"""Client implementation for the TMInterface ≥2.0 AngelScript bridge."""
from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional

import numpy as np


class MessageType(IntEnum):
    SC_RUN_STEP_SYNC = 1
    SC_CHECKPOINT_COUNT_CHANGED_SYNC = 2
    SC_LAP_COUNT_CHANGED_SYNC = 3
    SC_REQUESTED_FRAME_SYNC = 4
    SC_ON_CONNECT_SYNC = 5
    C_SET_SPEED = 6
    C_REWIND_TO_STATE = 7  # unused but kept for parity
    C_REWIND_TO_CURRENT_STATE = 8  # unused
    C_GET_SIMULATION_STATE = 9  # unused
    C_SET_INPUT_STATE = 10
    C_GIVE_UP = 11  # unused
    C_PREVENT_SIMULATION_FINISH = 12
    C_SHUTDOWN = 13
    C_EXECUTE_COMMAND = 14
    C_SET_TIMEOUT = 15  # unused
    C_RACE_FINISHED = 16  # unused
    C_REQUEST_FRAME = 17  # unused
    C_RESET_CAMERA = 18  # unused
    C_SET_ON_STEP_PERIOD = 19  # unused
    C_UNREQUEST_FRAME = 20  # unused
    C_TOGGLE_INTERFACE = 21  # unused
    C_IS_IN_MENUS = 22  # unused
    C_GET_INPUTS = 23  # unused


RUN_STEP_STRUCT = struct.Struct(
    "<i f f f f f f f f i i B B B B B B B I"
)


@dataclass
class TMIConfig:
    host: str = "127.0.0.1"
    port: int = 54540
    game_speed: float = 1.0
    prevent_finish: bool = True
    connect_timeout_s: float = 10.0
    server_name: str = "TMInterface0"  # unused, kept for backwards compatibility


class TMIClient:
    """Minimal client used by the RL environment."""

    def __init__(self, cfg: Optional[TMIConfig] = None):
        self.cfg = cfg or TMIConfig()
        self._sock: Optional[socket.socket] = None
        self._sock_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

        self._latest_state: Dict[str, float] = {}
        self._state_lock = threading.Lock()
        self._new_state_event = threading.Event()
        self._connected_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._sock is not None:
            return

        self._connected_event.clear()
        self._new_state_event.clear()

        self._sock = socket.create_connection(
            (self.cfg.host, self.cfg.port), timeout=self.cfg.connect_timeout_s
        )
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self._running = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="tmiface-plugin-reader", daemon=True
        )
        self._reader_thread.start()

        if not self._connected_event.wait(self.cfg.connect_timeout_s):
            self.close()
            raise TimeoutError(
                "Timed out while waiting for SC_ON_CONNECT from TMInterface plugin"
            )

        self.set_game_speed(self.cfg.game_speed)
        if self.cfg.prevent_finish:
            self.prevent_finish(True)

    def close(self) -> None:
        self._running = False
        try:
            if self._sock is not None:
                try:
                    self._send_int(MessageType.C_SHUTDOWN)
                except OSError:
                    pass
                self._sock.close()
        finally:
            self._sock = None
            if self._reader_thread is not None:
                self._reader_thread.join(timeout=1.0)
                self._reader_thread = None
            self._connected_event.set()
            self._new_state_event.set()

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    def is_connected(self) -> bool:
        return self._sock is not None and self._running

    def wait_for_state(self, timeout: float = 0.5) -> bool:
        got = self._new_state_event.wait(timeout)
        if got:
            self._new_state_event.clear()
        return got

    def get_state(self) -> Dict[str, float]:
        with self._state_lock:
            return dict(self._latest_state)

    def send_inputs(self, steer: float, gas: float, brake: bool) -> None:
        if self._sock is None:
            raise RuntimeError("TMInterface plugin is not connected")

        steer = float(np.clip(steer, -1.0, 1.0))
        gas = float(np.clip(gas, 0.0, 1.0))
        brake_b = bool(brake)
        accelerate = gas > 0.0

        steer_i = int(round(steer * 65536))
        gas_i = int(round(gas * 65536))
        payload = struct.pack(
            "<iiiBB",
            MessageType.C_SET_INPUT_STATE,
            steer_i,
            gas_i,
            int(accelerate),
            int(brake_b),
        )
        self._sendall(payload)

    def respawn(self, to_start: bool = False) -> None:
        command = "press delete" if to_start else "press enter"
        self.execute_command(command)

    def execute_command(self, command: str) -> None:
        if self._sock is None:
            return
        data = command.encode("utf-8")
        header = struct.pack("<ii", MessageType.C_EXECUTE_COMMAND, len(data))
        self._sendall(header + data)

    def set_game_speed(self, speed: Optional[float] = None) -> None:
        if self._sock is None:
            raise RuntimeError("TMInterface plugin is not connected")
        spd = self.cfg.game_speed if speed is None else float(speed)
        payload = struct.pack("<if", MessageType.C_SET_SPEED, spd)
        self._sendall(payload)

    def prevent_finish(self, enabled: bool = True) -> None:
        if not enabled or self._sock is None:
            return
        self._send_int(MessageType.C_PREVENT_SIMULATION_FINISH)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _reader_loop(self) -> None:
        try:
            while self._running and self._sock is not None:
                raw = self._recv_exact(4)
                if not raw:
                    break
                (msg_type,) = struct.unpack("<i", raw)
                if msg_type == MessageType.SC_RUN_STEP_SYNC:
                    payload = self._recv_exact(RUN_STEP_STRUCT.size)
                    if not payload:
                        break
                    state = self._decode_state(payload)
                    with self._state_lock:
                        self._latest_state = state
                    self._new_state_event.set()
                    self._send_int(MessageType.SC_RUN_STEP_SYNC)
                elif msg_type == MessageType.SC_ON_CONNECT_SYNC:
                    self._send_int(MessageType.SC_ON_CONNECT_SYNC)
                    self._connected_event.set()
                else:
                    # Acknowledge and ignore unhandled messages.
                    self._send_int(msg_type)
        except (OSError, struct.error):
            pass
        finally:
            self._running = False
            self._connected_event.set()
            self._new_state_event.set()

    def _recv_exact(self, size: int) -> bytes:
        if self._sock is None:
            return b""
        chunks = bytearray()
        remaining = size
        while remaining > 0:
            chunk = self._sock.recv(remaining)
            if not chunk:
                return b""
            chunks.extend(chunk)
            remaining -= len(chunk)
        return bytes(chunks)

    def _send_int(self, value: IntEnum) -> None:
        self._sendall(struct.pack("<i", int(value)))

    def _sendall(self, data: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("TMInterface plugin is not connected")
        with self._sock_lock:
            self._sock.sendall(data)

    @staticmethod
    def _decode_state(payload: bytes) -> Dict[str, float]:
        (
            race_time_ms,
            pos_x,
            pos_y,
            pos_z,
            yaw,
            speed,
            vel_x,
            vel_y,
            vel_z,
            cp_index,
            lap,
            lateral,
            finished,
            fl,
            fr,
            rl,
            rr,
            wheels_grounded,
            wall_count,
        ) = RUN_STEP_STRUCT.unpack(payload)

        wheel_contacts = {
            "front_left": bool(fl),
            "front_right": bool(fr),
            "rear_left": bool(rl),
            "rear_right": bool(rr),
        }

        return {
            "position": (pos_x, pos_y, pos_z),
            "yaw": yaw,
            "speed": speed,
            "velocity": (vel_x, vel_y, vel_z),
            "cp_index": int(cp_index),
            "lap": int(lap),
            "race_time_ms": int(race_time_ms),
            "race_time": float(race_time_ms) / 1000.0,
            "has_any_lateral_contact": bool(lateral),
            "finished": bool(finished),
            "wheel_ground_contact": wheel_contacts,
            "nb_wheels_grounded": int(wheels_grounded),
            "wall_contact_count": int(wall_count),
        }
