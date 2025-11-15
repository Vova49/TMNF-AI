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

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if not self.client.is_connected():
            self.client.start()
        else:
            self.client.respawn()

        # ждём первый тик от on_run_step, чтобы state был валидный
        if not self.client.wait_for_state(5):
            raise RuntimeError("TMInterface не прислал состояние в отведённое время")

        self._tick_counter = 0
        self._prev_state = self.client.get_state()
        self._current_state = self._prev_state

        # включаем кулдаун
        # reset cp-index tracker
        try:
            self._prev_cp_index = int(self._current_state.get('cp_index', 0)) if isinstance(self._current_state,
                                                                                            dict) else None
        except Exception:
            self._prev_cp_index = None

        # нейтральных инпутов после респавна
        self._cooldown_left_ticks = int(self.respawn_cooldown_ticks)
        if isinstance(self._current_state, dict):
            # флаг полезен для reward, чтобы игнорить отрицат. ds из-за телепорта
            self._current_state["just_respawned"] = True

        # print(f"[Env] reset: ticks=0, state received, respawn cooldown = {self._cooldown_left_ticks} ticks")
        obs = self._state_to_obs(self._current_state)
        return obs, {}

    # env.py — внутри TrackmaniaEnv.step
    def step(self, action: np.ndarray):
        steer = float(action[0])
        gas = float(action[1])
        brake = bool(action[2] > 0.5)

        cooldown_active = self._cooldown_left_ticks > 0
        # 1) Зафиксировать инпут (или нейтраль во время кулдауна)
        if cooldown_active:
            self.client.send_inputs(steer=0.0, gas=0.0, brake=False)
        else:
            self.client.send_inputs(steer=steer, gas=gas, brake=brake)

        total_reward = 0.0
        last_obs = None

        # 2) Дождаться и агрегировать ticks_per_step тиков физики
        ticks_to_wait = int(self.ticks_per_step)
        for _ in range(ticks_to_wait):
            # ждём следующий тик on_run_step
            if not self.client.wait_for_state(timeout=0.4):
                # нет нового тика — можно мягко продолжить или поднять warning
                pass

            # обновить состояния
            prev_state = self._current_state
            cur_state = self.client.get_state()
            self._prev_state = prev_state
            self._current_state = cur_state

            # помечаем телепорт/респаун во время кулдауна
            if cooldown_active and isinstance(cur_state, dict):
                cur_state["just_respawned"] = True

            # покадровая награда
            r = (
                rewards.compute_reward(prev_state, cur_state)
                if prev_state is not None else 0.0
            )
            total_reward += float(r)

            # наблюдение на основе последнего тика
            last_obs = self._state_to_obs(cur_state)

            # уменьшаем кулдаун по тикам
            if cooldown_active:
                self._cooldown_left_ticks = max(0, self._cooldown_left_ticks - 1)

        # учёт счётчика тиков и терминальности
        self._tick_counter += ticks_to_wait
        terminated = self._is_terminal(self._current_state)
        truncated = self._tick_counter >= self.episode_max_ticks
        info = {
            "last_inputs": (steer, gas, brake),
            "cooldown_active": cooldown_active,
            "cooldown_left_ticks": max(0, self._cooldown_left_ticks),
            "cp_index": int(self._current_state.get("cp_index", -1)) if isinstance(self._current_state, dict) else -1,
            "race_time": float(self._current_state.get("race_time", 0.0)) if isinstance(self._current_state,
                                                                                        dict) else 0.0,
        }

        # запасной вариант, если не пришло новое состояние
        if last_obs is None:
            last_obs = self._state_to_obs(self._current_state)

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
        yaw = float(state.get("yaw", 0.0))
        speed = float(state.get("speed", 0.0))

        # вычисляем геометрию относительно центра
        s, d, tangent_angle, dist_cp = self.center.project_with_extras(pos)
        ang_diff = self._angle_diff(yaw, tangent_angle)

        # Сохраняем производные величины в state, чтобы их можно логировать/использовать далее
        state["s"] = float(s)
        state["d"] = float(d)
        state["tangent_angle"] = float(tangent_angle)
        state["dist_cp"] = float(dist_cp)
        state["ang_diff"] = float(ang_diff)

        return np.array([speed, yaw, s, d, ang_diff, dist_cp], dtype=np.float32)

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
