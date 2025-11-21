# debug_rewards_live.py
"""
Онлайновый просмотр награды по текущей reward-функции во время ручного проезда.

Скрипт:
- подключается к TMInterface;
- на каждом тике читает состояние машины;
- проецирует позицию на центрлайн (как в TrackmaniaEnv);
- считает rewards.compute_reward(prev_state, cur_state, info);
- выводит всю информацию понятным текстом.

ВАЖНО: скрипт НЕ отправляет инпуты, управлять машиной можно с клавиатуры.
"""

from __future__ import annotations

import math
from typing import Dict, Any

from trackmania_rl import rewards, config as cfg
from trackmania_rl.track_pipeline import CenterLine
from trackmania_rl.tmi_client import TMIClient, TMIConfig


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def augment_state(raw_state: Dict[str, Any], center: CenterLine, heading: float | None = None) -> Dict[str, Any]:
    state: Dict[str, Any] = dict(raw_state)

    pos = state.get("position", (0.0, 0.0, 0.0))
    yaw_raw = float(state.get("yaw", 0.0))

    s, d, tangent_angle, dist_cp = center.project_with_extras(pos)

    yaw_trigo = yaw_raw - math.pi / 2.0
    yaw_trigo = ((yaw_trigo + math.pi) % (2.0 * math.pi)) - math.pi

    base_heading = heading if heading is not None else yaw_trigo
    ang_diff = _angle_diff(base_heading, tangent_angle)

    state["s"] = float(s)
    state["d"] = float(d)
    state["tangent_angle"] = float(tangent_angle)
    state["dist_cp"] = float(dist_cp)
    state["yaw_trigo"] = float(yaw_trigo)
    state["ang_diff"] = float(ang_diff)
    if heading is not None:
        state["heading"] = float(heading)

    return state



def main() -> None:
    # 1. Загружаем центрлайн так же, как это делает TrackmaniaEnv
    centerline_path = cfg.env.centerline_path
    center = CenterLine(centerline_path)

    # 2. Настройки подключения к TMInterface — как в TrackmaniaEnv
    tmi_cfg = TMIConfig(
        host=cfg.tmiface.host,
        port=cfg.tmiface.port,
        game_speed=cfg.tmiface.game_speed,
        prevent_finish=cfg.tmiface.prevent_finish,
        connect_timeout_s=cfg.tmiface.connect_timeout,
    )
    client = TMIClient(cfg=tmi_cfg)

    print(f"[Отладка] Подключение к TMInterface по адресу {tmi_cfg.host}:{tmi_cfg.port} ...")
    client.start()
    print("[Отладка] Подключено. Управляй машиной сам, скрипт только читает состояние.\n")

    prev_state: Dict[str, Any] | None = None
    step_idx: int = 0
    ep_reward: float = 0.0

    try:
        last_pos = None
        last_heading = 0.0
        while True:
            if not client.wait_for_state(timeout=1.0):
                continue

            raw = client.get_state()

            pos = raw.get("position", (0.0, 0.0, 0.0)) if isinstance(raw, dict) else (0.0, 0.0, 0.0)
            x, y, z = pos
            heading = float(last_heading)
            if last_pos is not None:
                dx = x - last_pos[0]
                dz = z - last_pos[2]
                if dx * dx + dz * dz > 1e-6:
                    heading = math.atan2(dz, dx)
            last_pos = (x, y, z)
            last_heading = heading

            cur_state = augment_state(raw, center, heading=heading)

            if prev_state is None:
                # Первый валидный state — используем как отправную точку
                prev_state = cur_state
                continue

            info: Dict[str, Any] = {}
            r = rewards.compute_reward(prev_state, cur_state, info)
            ep_reward += float(r)
            step_idx += 1

            # Достаём удобные значения для объяснения
            race_time = float(cur_state.get("race_time", 0.0))

            s = float(cur_state.get("s", 0.0))
            ds = float(info.get("ds", 0.0))
            d = float(cur_state.get("d", 0.0))

            cp_index = int(cur_state.get("cp_index", -1))
            speed = float(cur_state.get("speed", 0.0))

            wall_contact = bool(
                info.get("wall_contact", cur_state.get("has_any_lateral_contact", False))
            )

            ang_diff_rad = float(cur_state.get("ang_diff", 0.0))
            ang_diff_deg = ang_diff_rad * 180.0 / math.pi

            align = float(info.get("align", 0.0))         # cos угла
            align_eff = float(info.get("align_eff", 0.0))  # cos^gamma после обрезки

            r_progress = float(info.get("r_progress", 0.0))
            r_cp = float(info.get("r_cp", 0.0))
            r_cp_shap = float(info.get("r_cp_shap", 0.0))
            r_wall = float(info.get("r_wall", 0.0))
            r_idle = float(info.get("r_idle", 0.0))
            r_backward = float(info.get("r_backward", 0.0))
            r_smooth = float(info.get("r_smooth", 0.0))

            idle_thresh = float(getattr(cfg.reward, "IDLE_SPEED_THRESH", 15.0))
            idle_w = float(getattr(cfg.reward, "W_IDLE", 0.1))
            idle_active = speed < idle_thresh

            text_wall = "ЕСТЬ контакт со стеной" if wall_contact else "НЕТ контакта со стеной"

            # Печатаем подробное пояснение по шагу
            print(
                f"\n================ Шаг {step_idx} =================\n"
                f"Время в гонке: {race_time:.3f} секунд\n"
                f"Номер текущего чекпоинта (cp_index): {cp_index}\n"
                f"\n"
                f"Положение машины относительно трассы:\n"
                f"  • Прогресс вдоль трассы (s): {s:.2f} метров\n"
                f"  • Изменение прогресса за этот шаг (ds): {ds:.3f} метров\n"
                f"    (положительное значение — движение вперёд по направлению трассы)\n"
                f"  • Поперечное смещение от центра трассы (d): {d:.2f} метров\n"
                f"    (со знаком: положительное — влево, отрицательное — вправо)\n"
                f"  • Скорость: {speed:.1f} м/с\n"
                f"  • Угол машины относительно направления трассы (ang_diff): "
                f"{ang_diff_deg:.1f} градусов ({ang_diff_rad:.3f} радиан)\n"
                f"  • Коэффициент выравнивания по направлению трассы (cos угла): {align:.3f}\n"
                f"  • Эффективный коэффициент выравнивания после обрезки и степеней: {align_eff:.3f}\n"
                f"  • Контакт со стеной: {text_wall}\n"
                f"\n"
                f"Награда за этот шаг: {float(r):.4f}\n"
                f"Разложение награды по составляющим:\n"
                f"  • Награда за прогресс вдоль трассы: {r_progress:.4f}\n"
                f"  • Награда за взятие чекпоинта: {r_cp:.4f}\n"
                f"  • Награда за приближение к следующему чекпоинту (shaping по CP): {r_cp_shap:.4f}\n"
                f"  • Штраф за касание стены: {r_wall:.4f}\n"
                f"  • Штраф за низкую скорость (< {idle_thresh:.1f} м/с): {r_idle:.4f}"
                f" — {'применён' if idle_active else 'не применяется'} (вес {idle_w:.3f})\n"
                f"  • Штраф за движение назад по трассе: {r_backward:.4f}\n"
                f"  • Штраф/бонус за плавность поворота (по изменению угла): {r_smooth:.4f}\n"
                f"\n"
                f"Накопленная награда за весь текущий заезд (эпизод): {ep_reward:.4f}\n"
            )

            prev_state = cur_state

    except KeyboardInterrupt:
        print("\n[Отладка] Работа скрипта остановлена пользователем (Ctrl+C).")
    finally:
        client.close()
        print("[Отладка] Соединение с TMInterface закрыто.")


if __name__ == "__main__":
    main()
