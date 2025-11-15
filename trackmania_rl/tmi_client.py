# TMInterface client wrapper for TrackMania Nations Forever
# Provides get_state() and send_inputs() for RL environment
#
# Changes in this version:
# - "Neutral on idle" feature: if no inputs have been sent for N ticks, send neutral controls (0 steer, 0 gas, brake=False)
#   to unlock any "stuck key" state in the game. Configurable via TMIConfig(neutral_on_idle, idle_ticks_threshold)
# - Public helper configure_idle_neutral() to tweak behavior at runtime.

import math
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
from tminterface.client import Client
from tminterface.interface import TMInterface


@dataclass
class TMIConfig:
    server_name: str = "TMInterface0"
    game_speed: float = 1.0
    prevent_finish: bool = True
    connect_timeout_s: float = 10.0
    # --- NEW: neutral-on-idle control ---
    neutral_on_idle: bool = True  # включить отправку нейтральных инпутов при простое
    idle_ticks_threshold: int = 64  # сколько тиков простоя должно пройти, прежде чем слать нейтраль


class _RLClient(Client):
    def __init__(self, outer):
        self.outer = outer

    def on_registered(self, iface: TMInterface):
        self.outer._iface = iface
        self.outer._connected_event.set()
        iface.log("RL client connected")

        # стартовые настройки (безопасно оборачиваем)
        try:
            # опционально: увеличить окно ответа от клиента (мс), чтобы не ловить дерег из-за лагов
            iface.set_timeout(-1)
            iface.set_speed(self.outer.cfg.game_speed)
            if self.outer.cfg.prevent_finish:
                iface.prevent_simulation_finish()
        except Exception as e:
            iface.log(f"init settings error: {e}", "warning")

        iface.log("RL client connected")

    # ---- robust readers -------------------------------------------------

    @staticmethod
    def _get_xyz(pos) -> tuple[float, float, float]:
        # list / tuple / numpy
        if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 3:
            return float(pos[0]), float(pos[1]), float(pos[2])
        # object with attributes
        for xk, yk, zk in (("x", "y", "z"), ("X", "Y", "Z")):
            if hasattr(pos, xk) and hasattr(pos, yk) and hasattr(pos, zk):
                return float(getattr(pos, xk)), float(getattr(pos, yk)), float(getattr(pos, zk))
        raise TypeError(f"Unsupported position type: {type(pos)}")

    @staticmethod
    def _get_yaw(s) -> float:
        # предпочтительно yaw / yaw_angle
        for name in ("yaw", "yaw_angle"):
            if hasattr(s, name):
                return float(getattr(s, name))
        # из кватерниона/матрицы при необходимости (упрощённо опускаем для TMNF)
        return 0.0

    @staticmethod
    def _get_speed(s) -> float:
        # предпочтительно display_speed -> speed -> из вектора
        for name in ("display_speed", "speed"):
            if hasattr(s, name):
                return float(getattr(s, name))
        vel = getattr(s, "velocity", None)
        if isinstance(vel, (list, tuple, np.ndarray)) and len(vel) >= 3:
            return float(math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2))
        # fallback
        return 0.0

    # ---- main tick callback ---------------------------------------------
    def on_run_step(self, iface: TMInterface, _time: int):
        try:
            s = iface.get_simulation_state()

            # --- базовые признаки из симуляции ---
            px, py, pz = self._get_xyz(getattr(s, "position", [0.0, 0.0, 0.0]))
            yaw = self._get_yaw(s)
            speed = self._get_speed(s)

            # --- касание боковой поверхностью (стена/бордюр) ---
            wall_contact = False
            try:
                veh = getattr(s, "scene_mobil", None)  # SceneVehicleCar
                if veh is not None:
                    wall_contact = bool(getattr(veh, "has_any_lateral_contact", False))
            except Exception:
                wall_contact = False

            # --- лениво инициализируем счётчик/флаг на outer ---
            if not hasattr(self.outer, "_wall_contact_count"):
                self.outer._wall_contact_count = 0
            if not hasattr(self.outer, "_wall_prev"):
                self.outer._wall_prev = False

            # считаем фронты касаний внутри тиков TMInterface
            if wall_contact and not self.outer._wall_prev:
                self.outer._wall_contact_count += 1
            self.outer._wall_prev = wall_contact

            # --- контакт колес с поверхностью (ground contact) ---
            wheel_contacts = {}
            try:
                wheels = getattr(s, "simulation_wheels", None) or getattr(s, "wheels", None)
                # возможные имена колёс в API
                name_pairs = [
                    ("front_left", "front_right", "back_left", "back_right"),
                    ("front_left", "front_right", "rear_left", "rear_right"),
                ]
                chosen = None
                for names in name_pairs:
                    if wheels and all(hasattr(wheels, n) for n in names):
                        chosen = names
                        break
                if wheels and chosen:
                    for name in chosen:
                        try:
                            w = getattr(wheels, name)
                            rts = getattr(w, "real_time_state", w)
                            has_contact = getattr(rts, "has_ground_contact", None)
                            if has_contact is None:
                                has_contact = getattr(rts, "ground_contact", False)
                            wheel_contacts[name] = bool(has_contact)
                        except Exception:
                            wheel_contacts[name] = False
                else:
                    # если API не отдал колёса, оставим пусто
                    wheel_contacts = {}
            except Exception:
                wheel_contacts = {}

            # --- чекпоинт (с аккуратным фоллбеком) ---
            cp_idx = 0
            try:
                pinfo = getattr(s, "player_info", None)
                if pinfo is not None and hasattr(pinfo, "cur_checkpoint"):
                    cp_idx = int(getattr(pinfo, "cur_checkpoint"))
                else:
                    cp_idx = int(getattr(s, "cur_checkpoint", 0))
            except Exception:
                cp_idx = 0

            # --- формируем состояние для env/reward ---
            state = {
                "speed": float(speed),
                "yaw": float(yaw),
                "position": (float(px), float(py), float(pz)),

                # совместимость со старым кодом
                "progress": float(getattr(s, "race_progress", 0.0)),
                "checkpoint": int(getattr(s, "cur_checkpoint", cp_idx)),
                "time": float(getattr(s, "race_time", getattr(s, "time", 0.0))),

                # ключи, которые использует rewards.py
                "cp_index": int(cp_idx),
                "race_time": float(getattr(s, "race_time", getattr(s, "time", 0.0))),
                "has_any_lateral_contact": bool(wall_contact),

                # НОВОЕ: счётчик касаний стены (накапливается в тиках TMI)
                "wall_contact_count": int(self.outer._wall_contact_count),
                # НОВОЕ: наличие контакта колёс с землёй
                "wheel_ground_contact": dict(wheel_contacts),
                "nb_wheels_grounded": int(sum(1 for v in wheel_contacts.values() if v)),
            }

            # публикуем последнее состояние
            with self.outer._lock:
                self.outer._latest_state = state
            self.outer._new_state_event.set()

            # --- безопасная отправка pending-инпутов из того же потока TMInterface ---
            pending: Optional[Tuple[int, int, bool]] = None
            with self.outer._lock:
                if self.outer._pending_input is not None:
                    pending = self.outer._pending_input
                    self.outer._pending_input = None

            if pending is not None:
                steer_i, gas_i, brake_b = pending
                try:
                    iface.set_input_state(
                        sim_clear_buffer=True,
                        steer=steer_i,
                        gas=gas_i,
                        brake=brake_b,
                        accelerate=(gas_i > 0),
                    )
                    # обновляем счётчики «простоя» и последнее отправленное
                    self.outer._ticks_since_input = 0
                    self.outer._last_sent_input = (steer_i, gas_i, brake_b)
                    self.outer._idle_neutral_latched = False
                except Exception as e:
                    try:
                        iface.log(f"set_input_state failed: {e}")
                    except Exception:
                        pass
                finally:
                    # гарантированно сбросили pending
                    with self.outer._lock:
                        if self.outer._pending_input == pending:
                            self.outer._pending_input = None

            else:
                # если инпутов нет — считаем тики простоя и при необходимости шлём «нейтраль»
                self.outer._ticks_since_input += 1
                if getattr(self.outer.cfg, "neutral_on_idle", False) and \
                        self.outer._ticks_since_input >= max(1,
                                                             int(getattr(self.outer.cfg, "idle_ticks_threshold", 10))):

                    if (not self.outer._idle_neutral_latched) or (self.outer._last_sent_input != (0, 0, False)):
                        try:
                            iface.set_input_state(
                                sim_clear_buffer=True,
                                steer=0,
                                gas=0,
                                brake=False,
                                accelerate=False,
                            )
                            self.outer._last_sent_input = (0, 0, False)
                            self.outer._idle_neutral_latched = True
                        except Exception as e:
                            try:
                                iface.log(f"neutral_on_idle failed: {e}")
                            except Exception:
                                pass

        except Exception as e:
            # не уронить клиент из-за единичной ошибки
            try:
                iface.log(f"on_run_step error: {e}")
            except Exception:
                pass

    def on_shutdown(self, iface: TMInterface):
        try:
            iface.log("RL client shutting down")
        except Exception:
            pass


class TMIClient:
    def __init__(self, cfg: Optional[TMIConfig] = None):
        self.cfg = cfg or TMIConfig()
        self._iface: Optional[TMInterface] = None
        self._connected_event = threading.Event()
        self._latest_state: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._new_state_event = threading.Event()
        self._pending_input: Optional[tuple[int, int, bool]] = None  # (steer_i, gas_i, brake_b)

        # --- idle control state ---
        self._ticks_since_input: int = 0
        self._idle_neutral_latched: bool = False
        self._last_sent_input: Tuple[int, int, bool] = (0, 0, False)  # last raw ints sent to iface

        self._wall_contact_count = 0
        self._wall_prev = False

    def start(self):
        # сохраняем ссылку, регистрируем клиента
        self._iface = TMInterface(self.cfg.server_name)
        print(f"[TMIClient] Подключаемся к серверу TMInterface: {self.cfg.server_name}...")
        self._client = _RLClient(self)
        ok = self._iface.register(self._client)
        if not ok:
            raise RuntimeError("TMInterface: клиент уже зарегистрирован")
        if not self._connected_event.wait(timeout=self.cfg.connect_timeout_s):
            raise TimeoutError("Не дождались on_registered от TMInterface")
        print("[TMIClient] Соединение с TMInterface установлено")

    def on_registered(self, iface: TMInterface):
        """Callback после регистрации TMInterface."""
        print("TMInterface зарегистрирован")
        self._iface = iface
        self._connected_event.set()

        # Настройка игры: скорость и запрет финиша
        try:
            self.set_game_speed(self.cfg.game_speed)
            if self.cfg.prevent_finish:
                self.prevent_finish(True)
        except Exception as e:
            print(f"Не удалось применить настройки TMInterface: {e}")

    def shutdown(self):
        try:
            if self._iface:
                self._iface.unregister()
        finally:
            self._iface = None

    def get_state(self) -> Dict[str, Any]:
        """Возвращает последнее состояние машины."""
        with self._lock:
            return dict(self._latest_state)

    # Управляющие команды

    def send_inputs(self, steer: float, gas: float, brake: float):
        # Нормируем входы ИИ
        steer = float(np.clip(steer, -1.0, 1.0))
        gas = float(np.clip(gas, 0.0, 1.0))
        brake_b = bool(brake >= 0.5)

        # Перевод в "сырые" int-диапазоны TMInterface
        steer_i = int(round(steer * 65536))  # [-65536..65536]
        gas_i = int(round(gas * 65536))  # [0..65536]

        # Не трогаем iface из внешнего потока — только кладём в pending
        with self._lock:
            self._pending_input = (steer_i, gas_i, brake_b)
        # Зафиксируем, что пришёл новый инпут (сброс простоев произойдёт в on_run_step после реальной отправки)

    # --- NEW: public helper to adjust idle behavior at runtime
    def configure_idle_neutral(self, enabled: bool, ticks: Optional[int] = None):
        """Включить/выключить и при необходимости изменить порог тиков для режима "нейтраль при простое".

        Args:
            enabled: включить/выключить поведение
            ticks: новый порог тиков (если None — оставить текущий)
        """
        self.cfg.neutral_on_idle = bool(enabled)
        if ticks is not None:
            self.cfg.idle_ticks_threshold = int(max(1, ticks))

    def set_game_speed(self, speed: float = None):
        """Установка скорости симуляции (game speed)."""
        if not self._iface:
            raise RuntimeError("TMInterface не подключен")
        speed = self.cfg.game_speed if speed is None else speed
        # TMInterface API: set_speed
        self._iface.set_speed(speed)
        print(f"Game speed установлен: {speed}")

    def prevent_finish(self, enabled: bool = True):
        if not self._iface:
            raise RuntimeError("TMInterface не подключен")
        if enabled:
            self._iface.prevent_simulation_finish()

    def wait_for_state(self, timeout: float = 0.2) -> bool:
        got = self._new_state_event.wait(timeout)
        self._new_state_event.clear()
        return got

    # Дополнительно: простые статусы/действия окружения
    def is_connected(self) -> bool:
        """Возвращает True, если TMInterface подключен."""
        return self._iface is not None

    def respawn(self, to_start: bool = False):
        """Респаун через консоль TMInterface.
        to_start=False — стандартный respawn на чекпоинт ("press enter").
        to_start=True  — полный рестарт ("press delete").
        """
        try:
            if self._iface is None:
                print("[TMI] respawn requested but iface is None")
                return
            key = "delete"
            # Всегда перезапуск через delete (рестарт гонки) по требованию пользователя
            self._iface.execute_command(f"press {key}")
            try:
                self._iface.log(f"respawn via {key}")
            except Exception:
                pass
            print("Произведен respawn")
        except Exception as e:
            print(f"Не удалось выполнить консольную команду respawn: {e}")
