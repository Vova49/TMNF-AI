from tminterface.client import Client, run_client
from tminterface.interface import TMInterface
import math


class DamperAbsorbClient(Client):
    def __init__(self):
        self._prev_vals = None  # last printed wheel values

    def on_registered(self, iface: TMInterface):
        try:
            iface.set_timeout(500)          # окно ответа клиента (мс)
            iface.set_speed(1.0)            # 1.0 = realtime
            iface.prevent_simulation_finish()
        except Exception as e:
            iface.log(f"[DamperAbsorb] init settings error: {e}", "warning")
        iface.log("[DamperAbsorb] client connected")

    def on_run_step(self, iface: TMInterface, _time: int):
        # Вызывается каждый физический тик
        try:
            s = iface.get_simulation_state()

            wheels = getattr(s, "simulation_wheels", None)  # np.ndarray из SimulationWheel
            n = int(len(wheels)) if wheels is not None else 0
            vals = []
            for i in range(n):
                wh = wheels[i]
                rt = getattr(wh, "real_time_state", None)
                val = float(getattr(rt, "damper_absorb", float("nan"))) if rt is not None else float("nan")
                vals.append(val)

            # Печатаем, только если есть изменения (чтобы не спамить)
            if self._should_log(vals):
                # форматирование: w0=..., w1=..., ...
                parts = [f"w{i}={('nan' if math.isnan(v) else f'{v:.6f}')}" for i, v in enumerate(vals)]
                finite = [v for v in vals if not math.isnan(v)]
                tail = f" | max={max(finite):.6f}" if finite else ""
                iface.log("[DamperAbsorb] " + ", ".join(parts) + tail)
                self._prev_vals = vals

        except Exception as e:
            iface.log(f"[DamperAbsorb] read error: {e}", "warning")

    def _should_log(self, vals, eps: float = 1e-4) -> bool:
        if self._prev_vals is None or len(self._prev_vals) != len(vals):
            return True
        for a, b in zip(self._prev_vals, vals):
            # любое NaN/число или изменение > eps — логируем
            if (math.isnan(a) != math.isnan(b)) or (not math.isnan(a) and abs(a - b) > eps):
                return True
        return False

    def on_shutdown(self, iface: TMInterface):
        iface.log("[DamperAbsorb] client disconnected")


if __name__ == "__main__":
    run_client(DamperAbsorbClient(), server_name="TMInterface0")
