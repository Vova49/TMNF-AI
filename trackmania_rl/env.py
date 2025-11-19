# env.py
"""Gymnasium environment for TrackMania Nations Forever (TMNF) using TMInterface.

Features:
1. Connection to the game via TMIClient.
2. Observations: speed, yaw, progress s along center line, lateral offset d,
   yaw diff to tangent, distance to next checkpoint (stub).
3. Action space: steer [-1,1], gas [0,1], brake {0,1}.
4. Aggregate several physics ticks per RL step (`ticks_per_step`).
5. Reward computed in `rewards.compute_reward`.
6. Respawn cooldown: после (ре)спавна несколько тиков шлём нейтральные инпуты.
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple
import math
import gymnasium as gym  # type: ignore
import numpy as np

from trackmania_rl import config as cfg
from trackmania_rl.tmi_client import TMIClient

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from trackmania_rl import track_pipeline as track
    from trackmania_rl import rewards
except ImportError:
    # fallback-заглушки, если модулей нет
    class rewards:  # type: ignore
        @staticmethod
        def compute_reward(prev_state, cur_state):  # noqa: D401
            return 0.0


    class track:  # type: ignore
        CenterLine = lambda path: None  # type: ignore


class TrackmaniaEnv(gym.Env):
    """OpenAI Gymnasium-compatible environment for TMNF."""

    metadata = {"render_modes": []}

    def __init__(
            self,
            centerline_path: str | None = None,
            ticks_per_step: int | None = None,
            episode_max_ticks: int | None = None,
            server_name: str | None = None,
    ) -> None:
        super().__init__()
        self.ticks_per_step = ticks_per_step or cfg.TICKS_PER_STEP
        self.episode_max_ticks = episode_max_ticks or cfg.EPISODE_MAX_TICKS
        self._tick_counter = 0

        # Respawn cooldown (в тиках физики). Можно задать в cfg.env.respawn_cooldown_ticks
        self.respawn_cooldown_ticks: int = int(
            getattr(getattr(cfg, "env", object()), "respawn_cooldown_ticks", 12)
        )
        self._cooldown_left_ticks: int = 0

        # Center line
        path = centerline_path or cfg.env.centerline_path
        self.center = track.CenterLine(path) if callable(getattr(track, "CenterLine", None)) else MinimalCenterLine()

        # Observation: [speed, yaw, s, d, ang_diff, dist_cp]
        low = np.array([0.0, -np.pi, 0.0, -50.0, -np.pi, 0.0], dtype=np.float32)
        high = np.array(
            [
                400.0,
                np.pi,
                getattr(self.center, "length", 10_000.0),
                50.0,
                np.pi,
                200.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Action: steer, gas, brake
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # TMI client
        from trackmania_rl.tmi_client import TMIConfig

        tmi_cfg = TMIConfig(
            host=server_name or getattr(cfg.tmiface, "host", "127.0.0.1"),
            port=getattr(cfg.tmiface, "port", 54540),
            game_speed=cfg.tmiface.game_speed,
            prevent_finish=cfg.tmiface.prevent_finish,
            connect_timeout_s=cfg.tmiface.connect_timeout,
        )
        self.client = TMIClient(cfg=tmi_cfg)

        self._prev_state: Dict[str, float] | None = None
        self._current_state: Dict[str, float] | None = None

        # internal: track previous checkpoint index for finish detection
        self._prev_cp_index: int | None = None

        # Для оценки heading по смещению в XZ
        self._last_pos: Tuple[float, float, float] | None = None
        self._last_heading: float = 0.0


    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if not self.client.is_connected():
            self.client.start()
        else:
            self.client.respawn(to_start=True)  # всегда Delete

        # ждём первый тик от on_run_step, чтобы state был валидный
        if not self.client.wait_for_state(5):
            raise RuntimeError("TMInterface не прислал состояние в отведённое время")

        self._tick_counter = 0
        self._prev_state = self.client.get_state()
        self._current_state = self._prev_state

        # reset cp-index tracker
        try:
            self._prev_cp_index = int(self._current_state.get("cp_index", 0)) \
                if isinstance(self._current_state, dict) else None
        except Exception:
            self._prev_cp_index = None

        # сбрасываем историю позиции/направления
        if isinstance(self._current_state, dict):
            # если в стейте нет поля "position", возьмётся (0, 0, 0)
            self._last_pos = self._current_state.get("position", (0.0, 0.0, 0.0))
        else:
            self._last_pos = None
        self._last_heading = 0.0

        # нейтральных инпутов после респавна
        self._cooldown_left_ticks = int(self.respawn_cooldown_ticks)
        if isinstance(self._current_state, dict):
            # флаг полезен для reward, чтобы игнорить отрицат. ds из-за телепорта
            self._current_state["just_respawned"] = True

        obs = self._state_to_obs(self._current_state)
        return obs, {}

    # env.py — внутри TrackmaniaEnv.step
    def step(self, action: np.ndarray):
        """Один RL-шаг: (steer, gas, brake) -> наблюдение, награда и флаги done."""
        steer = float(action[0])
        gas = float(action[1])
        brake = bool(action[2] > 0.5)

        cooldown_active = self._cooldown_left_ticks > 0

        # Команды, которые реально ушли в игру
        steer_cmd = 0.0 if cooldown_active else steer
        gas_cmd = 0.0 if cooldown_active else gas
        brake_cmd = False if cooldown_active else brake

        # Отправляем инпут один раз в начале RL-шага
        self.client.send_inputs(steer=steer_cmd, gas=gas_cmd, brake=brake_cmd)

        total_reward = 0.0
        last_obs = None

        ticks_to_wait = int(self.ticks_per_step)
        for _ in range(ticks_to_wait):
            # ждём следующий тик on_run_step
            if not self.client.wait_for_state(timeout=0.4):
                # нет нового тика — используем последнее состояние
                pass

            prev_state = self._current_state
            cur_state = self.client.get_state()
            self._prev_state = prev_state
            self._current_state = cur_state

            if not isinstance(cur_state, dict):
                # защитимся от неожиданностей
                cur_state = {}
                self._current_state = cur_state

            # Помечаем телепорт/респавн во время кулдауна
            if cooldown_active:
                cur_state["just_respawned"] = True

            # Прокидываем реальные команды управления в state
            cur_state["cmd_steer"] = float(steer_cmd)
            cur_state["cmd_gas"] = float(gas_cmd)
            cur_state["cmd_brake"] = bool(brake_cmd)

            # СНАЧАЛА считаем геометрию относительно центра (s, d, ang_diff, dist_cp...)
            obs_tick = self._state_to_obs(cur_state)

            # ПОТОМ считаем награду — cur_state уже обогащён производными величинами
            r = (
                rewards.compute_reward(prev_state, cur_state)
                if prev_state is not None
                else 0.0
            )
            total_reward += float(r)

            last_obs = obs_tick

            # уменьшаем кулдаун по тикам
            if cooldown_active and self._cooldown_left_ticks > 0:
                self._cooldown_left_ticks = max(0, self._cooldown_left_ticks - 1)

        # учёт счётчика тиков и терминальности
        self._tick_counter += ticks_to_wait
        cur_state = self._current_state if isinstance(self._current_state, dict) else {}

        terminated = self._is_terminal(cur_state)
        truncated = self._tick_counter >= self.episode_max_ticks

        info = {
            "last_inputs": (steer_cmd, gas_cmd, brake_cmd),
            "cooldown_active": cooldown_active,
            "cooldown_left_ticks": max(0, self._cooldown_left_ticks),
            "cp_index": int(cur_state.get("cp_index", -1)),
            "race_time": float(cur_state.get("race_time", 0.0)),
        }

        # запасной вариант, если по какой-то причине obs_tick не обновился
        if last_obs is None:
            last_obs = self._state_to_obs(cur_state)

        return last_obs, total_reward, terminated, truncated, info

    def render(self):  # noqa: D401
        return None  # no-op for now

    def close(self):
        self.client.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _state_to_obs(self, state: Dict[str, float]):
        pos = state.get("position", (0.0, 0.0, 0.0))
        speed = float(state.get("speed", 0.0))

        # --- 1) Оценка heading по смещению в XZ ---
        x, y, z = pos
        heading = float(self._last_heading)

        if self._last_pos is not None:
            dx = x - self._last_pos[0]
            dz = z - self._last_pos[2]
            if dx * dx + dz * dz > 1e-6:
                heading = math.atan2(dz, dx)

        self._last_pos = (x, y, z)
        self._last_heading = heading

        # --- 2) Геометрия относительно центра ---
        s, d, tangent_angle, dist_cp = self.center.project_with_extras(pos)
        ang_diff = self._angle_diff(heading, tangent_angle)

        # --- 3) Сохраняем производные величины в state ---
        state["s"] = float(s)
        state["d"] = float(d)
        state["tangent_angle"] = float(tangent_angle)
        state["dist_cp"] = float(dist_cp)
        state["ang_diff"] = float(ang_diff)

        # полезно сохранить и сырые yaw из TMI, если захочешь отладить
        state["heading"] = float(heading)
        state["yaw_raw"] = float(state.get("yaw", 0.0))

        # В obs вместо yaw теперь отдаём heading
        return np.array([speed, heading, s, d, ang_diff, dist_cp], dtype=np.float32)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return (a - b + np.pi) % (2 * np.pi) - np.pi

    def _is_terminal(self, state: Dict[str, float]):
        # Episode termination logic.
        #
        # Prefers explicit flags if they are present in *state*, but also supports
        # optional heuristics without touching the TMInterface client:
        #
        # 1) If state['finished'] is True — terminate.
        # 2) If cfg.env.finish_cp_index exists — terminate when cp_index >= that value.
        # 3) If prevent_finish is False and cp_index wrapped around (decreased) without a respawn —
        #    assume we crossed finish.
        # 4) Если машина упала ниже заданной высоты (Y < death_y_threshold) — считаем эпизод бесполезным и завершаем.

        # 1) explicit flag
        try:
            if bool(state.get("finished", False)):
                return True
        except Exception:
            pass

        # 2) optional config-driven target cp index
        try:
            target_cp = getattr(getattr(cfg, "env", object()), "finish_cp_index", None)
            if target_cp is not None:
                cp_now = int(state.get("cp_index", -1))
                if cp_now >= int(target_cp):
                    return True
        except Exception:
            pass

        # 3) checkpoint wrap-around heuristic (only meaningful if finish is not prevented)
        try:
            prevent_finish = bool(getattr(getattr(cfg, "tmiface", object()), "prevent_finish", True))
        except Exception:
            prevent_finish = True

        cp_now = None
        try:
            cp_now = int(state.get("cp_index"))
        except Exception:
            cp_now = None

        if not prevent_finish and cp_now is not None:
            prev = getattr(self, "_prev_cp_index", None)
            just_respawned = bool(state.get("just_respawned", False))
            # If cp index decreased and we didn't respawn — likely crossed finish line.
            if prev is not None and cp_now < prev and not just_respawned:
                return True

            # update tracker for next call
            self._prev_cp_index = cp_now

        # 4) падение ниже минимальной высоты Y
        try:
            pos = state.get("position", None)
            if pos is not None and len(pos) >= 2:
                y = float(pos[1])
                death_y = float(
                    getattr(getattr(cfg, "env", object()), "death_y_threshold", 15.0)
                )
                if y < death_y:
                    return True
        except Exception:
            pass

        return False


# ----------------------------------------------------------------------
# Minimal center-line implementation if track_pipeline missing
# ----------------------------------------------------------------------
class MinimalCenterLine:
    def __init__(self):
        self.points = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        self._s = np.array([0.0, 100.0])
        self.length = 100.0

    def project(self, pos: Tuple[float, float, float]):
        # naive projection onto straight line x-axis
        x, y, z = pos
        return x, np.hypot(y, z)

    def tangent_angle(self, s):
        return 0.0

    def dist_to_next_checkpoint(self, s, next_cp_s):
        return max(next_cp_s - s, 0.0)

    def project_with_extras(self, pos):
        s, d = self.project(pos)
        return s, d, 0.0, 0.0
