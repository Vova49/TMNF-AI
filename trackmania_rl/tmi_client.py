"""AngelScript bridge client for TMInterface 2.x.

This module replaces the legacy ``tminterface`` Python bindings with a small
TCP-based protocol that talks to an AngelScript plugin running inside
TMInterface.  The public API intentionally mirrors the previous implementation
so higher level RL code can stay untouched.
"""
from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class TMIConfig:
    """Configuration of the Python <-> AngelScript bridge."""

    server_name: str = "127.0.0.1"  # Host/IP that the AngelScript plugin connects to
    port: int = 54545  # TCP port listened by Python
    connect_timeout_s: float = 10.0
    game_speed: float = 1.0
    prevent_finish: bool = True
    neutral_on_idle: bool = True
    idle_ticks_threshold: int = 64


class TMIClient:
    """Socket-based client that exchanges data with the AngelScript plugin."""

    def __init__(self, cfg: Optional[TMIConfig] = None):
        self.cfg = cfg or TMIConfig()

        self._server_socket: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._connected_event = threading.Event()
        self._new_state_event = threading.Event()

        self._latest_state: Dict[str, Any] = {}
        self._running = False

        # state bookkeeping
        self._wall_contact_count = 0
        self._wall_prev = False
        self._ticks_since_input: int = 0
        self._idle_neutral_latched: bool = False
        self._last_sent_input: Tuple[float, float, bool] = (0.0, 0.0, False)

        self._recv_buffer = ""

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start listening for the AngelScript plugin and block until it connects."""
        if self._running:
            return

        self._prepare_listener()
        self._accept_connection()

        self._running = True
        self._connected_event.set()

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        # Apply initial runtime settings (mirrors legacy behaviour)
        try:
            self.set_game_speed(self.cfg.game_speed)
            if self.cfg.prevent_finish:
                self.prevent_finish(True)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[TMIClient] Failed to apply initial settings: {exc}")

        print("[TMIClient] AngelScript bridge connected")

    def shutdown(self) -> None:
        """Close the bridge connection and stop background threads."""
        self._running = False
        self._connected_event.clear()

        try:
            if self._conn is not None:
                try:
                    self._conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._conn.close()
        finally:
            self._conn = None

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            finally:
                self._server_socket = None

    # Legacy alias for environments expecting ``close``.
    close = shutdown

    def _prepare_listener(self) -> None:
        host = self.cfg.server_name
        port = int(self.cfg.port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1)
        self._server_socket = sock
        print(f"[TMIClient] Listening for TMInterface plugin on {host}:{port} ...")

    def _accept_connection(self) -> None:
        assert self._server_socket is not None
        self._server_socket.settimeout(self.cfg.connect_timeout_s)
        try:
            conn, addr = self._server_socket.accept()
        except socket.timeout as exc:
            raise TimeoutError("Timed out waiting for TMInterface plugin to connect") from exc

        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(None)
        self._conn = conn
        print(f"[TMIClient] Plugin connected from {addr}")

    # ------------------------------------------------------------------
    # Public API consumed by the Gym environment
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_state)

    def send_inputs(self, steer: float, gas: float, brake: float) -> None:
        steer = float(np.clip(steer, -1.0, 1.0))
        gas = float(np.clip(gas, 0.0, 1.0))
        brake_b = bool(brake >= 0.5)

        payload = {
            "type": "inputs",
            "steer": steer,
            "gas": gas,
            "brake": brake_b,
        }
        self._send_json(payload)

        self._ticks_since_input = 0
        self._idle_neutral_latched = False
        self._last_sent_input = (steer, gas, brake_b)

    def wait_for_state(self, timeout: float = 0.2) -> bool:
        got = self._new_state_event.wait(timeout)
        if got:
            self._new_state_event.clear()
        return got

    def is_connected(self) -> bool:
        return self._running and self._conn is not None

    def respawn(self, to_start: bool = False) -> None:
        self._send_command("respawn", to_start=bool(to_start))
        # Reset idle bookkeeping so we do not instantly neutralise
        self._ticks_since_input = 0
        self._idle_neutral_latched = False

    def set_game_speed(self, speed: Optional[float] = None) -> None:
        speed = float(self.cfg.game_speed if speed is None else speed)
        self.cfg.game_speed = speed
        self._send_command("set_game_speed", value=speed)

    def prevent_finish(self, enabled: bool = True) -> None:
        self._send_command("prevent_finish", value=bool(enabled))

    def configure_idle_neutral(self, enabled: bool, ticks: Optional[int] = None) -> None:
        self.cfg.neutral_on_idle = bool(enabled)
        if ticks is not None:
            self.cfg.idle_ticks_threshold = int(max(1, ticks))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _send_command(self, name: str, value: Any | None = None, **extra: Any) -> None:
        payload: Dict[str, Any] = {"type": "command", "name": name}
        if value is not None:
            payload["value"] = value
        if extra:
            payload.update(extra)
        self._send_json(payload)

    def _send_json(self, message: Dict[str, Any]) -> None:
        if self._conn is None:
            return
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        with self._send_lock:
            try:
                self._conn.sendall(data)
            except OSError:
                self._handle_disconnect()

    def _recv_loop(self) -> None:
        assert self._conn is not None
        conn = self._conn
        while self._running:
            try:
                chunk = conn.recv(4096)
            except OSError:
                self._handle_disconnect()
                break

            if not chunk:
                self._handle_disconnect()
                break

            self._recv_buffer += chunk.decode("utf-8", errors="ignore")
            self._drain_buffer()

    def _drain_buffer(self) -> None:
        while True:
            newline = self._recv_buffer.find("\n")
            if newline == -1:
                break
            line = self._recv_buffer[:newline]
            self._recv_buffer = self._recv_buffer[newline + 1:]
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[TMIClient] Failed to parse message: {exc}: {line!r}")
                continue
            self._handle_message(message)

    def _handle_message(self, message: Dict[str, Any]) -> None:
        msg_type = message.get("type")
        if msg_type == "state":
            state_payload = message.get("state")
            if state_payload is None:
                state_payload = {k: v for k, v in message.items() if k != "type"}
            self._handle_state(state_payload)
        elif msg_type == "event":
            name = message.get("name")
            if name == "respawn":
                self._ticks_since_input = 0
                self._idle_neutral_latched = False
        elif msg_type == "log":
            text = message.get("message", "")
            if text:
                print(f"[TMInterface] {text}")
        elif msg_type == "pong":
            pass
        else:
            # Assume it is already a state payload when no explicit type is supplied.
            if "speed" in message or "position" in message:
                self._handle_state(message)

    def _handle_state(self, payload: Dict[str, Any]) -> None:
        state = self._normalize_state(payload)
        with self._lock:
            self._latest_state = state
        self._new_state_event.set()

        # Idle neutral watchdog – executed on every physics tick
        self._ticks_since_input += 1
        if (
            self.cfg.neutral_on_idle
            and not self._idle_neutral_latched
            and self._ticks_since_input >= max(1, int(self.cfg.idle_ticks_threshold))
            and self._last_sent_input != (0.0, 0.0, False)
        ):
            self._send_neutral_inputs()

    def _send_neutral_inputs(self) -> None:
        self._send_json({"type": "inputs", "steer": 0.0, "gas": 0.0, "brake": False})
        self._idle_neutral_latched = True
        self._last_sent_input = (0.0, 0.0, False)
        self._ticks_since_input = 0

    def _normalize_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state: Dict[str, Any] = {}

        speed = payload.get("speed")
        if speed is None:
            speed = payload.get("display_speed")
        if speed is None and isinstance(payload.get("velocity"), (list, tuple)):
            vx, vy, vz = payload.get("velocity", (0.0, 0.0, 0.0))[:3]
            speed = float(np.linalg.norm([vx, vy, vz]))
        state["speed"] = float(speed or 0.0)

        yaw = payload.get("yaw")
        if yaw is None:
            yaw = payload.get("yaw_angle", 0.0)
        state["yaw"] = float(yaw or 0.0)

        state["position"] = self._extract_position(payload.get("position"))

        state["progress"] = float(payload.get("progress", 0.0))

        cp_index = int(payload.get("cp_index", payload.get("checkpoint", 0)))
        state["cp_index"] = cp_index
        state["checkpoint"] = int(payload.get("checkpoint", cp_index))

        race_time = float(payload.get("race_time", payload.get("time", 0.0)))
        state["race_time"] = race_time
        state["time"] = float(payload.get("time", race_time))

        has_wall_contact = bool(payload.get("has_any_lateral_contact", False))
        state["has_any_lateral_contact"] = has_wall_contact
        if has_wall_contact and not self._wall_prev:
            self._wall_contact_count += 1
        self._wall_prev = has_wall_contact
        state["wall_contact_count"] = self._wall_contact_count

        wheel_contacts = self._extract_wheel_contacts(payload.get("wheel_ground_contact"))
        state["wheel_ground_contact"] = wheel_contacts
        state["nb_wheels_grounded"] = int(sum(1 for v in wheel_contacts.values() if v))

        # Pass through optional fields if the plugin provides them
        for key in (
            "velocity",
            "curvature",
            "dist_cp",
            "tangent_angle",
            "just_respawned",
            "nb_wheels_grounded",
        ):
            if key in payload:
                state[key] = payload[key]

        return state

    @staticmethod
    def _extract_position(value: Any) -> Tuple[float, float, float]:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return float(value[0]), float(value[1]), float(value[2])
        if isinstance(value, dict):
            for keys in (("x", "y", "z"), ("X", "Y", "Z")):
                if all(k in value for k in keys):
                    return float(value[keys[0]]), float(value[keys[1]]), float(value[keys[2]])
        return 0.0, 0.0, 0.0

    @staticmethod
    def _extract_wheel_contacts(value: Any) -> Dict[str, bool]:
        if isinstance(value, dict):
            return {str(k): bool(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            result: Dict[str, bool] = {}
            for idx, item in enumerate(value):
                result[f"wheel_{idx}"] = bool(item)
            return result
        return {}

    def _handle_disconnect(self) -> None:
        if not self._running:
            return
        print("[TMIClient] Disconnected from AngelScript plugin")
        self.shutdown()

    # Context manager sugar -------------------------------------------------
    def __enter__(self) -> "TMIClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context manager helper
        self.shutdown()


__all__ = ["TMIConfig", "TMIClient"]
