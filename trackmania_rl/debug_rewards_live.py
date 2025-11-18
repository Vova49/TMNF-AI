from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Tuple

import numpy as np

# Импорты такие же, что и в env.py
from trackmania_rl import config as cfg
from trackmania_rl import track_pipeline as track
from trackmania_rl.tmi_client import TMIClient, TMIConfig


def angle_diff(a: float, b: float) -> float:
    """Нормализованная разница углов в радианах в диапазон [-pi, pi]."""
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def project_with_debug(
    center: track.CenterLine,
    pos: Tuple[float, float, float],
) -> tuple[float, float, float, int, float, float]:
    """
    Копия логики CenterLine.project + явный индекс сегмента и параметр t.

    Возвращает:
      s, d, tangent_angle, seg_idx, t, min_dist
    """
    px, _, pz = pos
    point = np.array([px, pz], dtype=np.float32)

    # Внутренние поля centerline (для дебага это ок)
    points = center._points          # shape (N, 3)
    s_arr = center._s                # shape (N,)
    seg_dirs = center._seg_dirs      # shape (N-1, 2)
    seg_angles = center._seg_angles  # shape (N-1,)

    min_dist = float("inf")
    best_idx = 0
    best_t = 0.0

    for i in range(len(points) - 1):
        a = points[i, [0, 2]]
        b = points[i + 1, [0, 2]]
        seg_vec = b - a
        seg_len_sq = float(np.dot(seg_vec, seg_vec))
        if seg_len_sq == 0.0:
            continue

        t = float(
            np.clip(
                np.dot(point - a, seg_vec) / seg_len_sq,
                0.0,
                1.0,
            )
        )
        proj = a + t * seg_vec
        dist = float(np.linalg.norm(point - proj))
        if dist < min_dist:
            min_dist = dist
            best_idx = i
            best_t = t

    # s по дуге
    s = float(s_arr[best_idx] + best_t * (s_arr[best_idx + 1] - s_arr[best_idx]))

    # знак бокового отклонения d через псевдо-кросс
    seg_dir = seg_dirs[best_idx]
    rel = point - points[best_idx, [0, 2]]
    cross = float(seg_dir[0] * rel[1] - seg_dir[1] * rel[0])
    d = float(min_dist * (1.0 if cross >= 0.0 else -1.0))

    tangent_angle = float(seg_angles[best_idx])

    return s, d, tangent_angle, best_idx, best_t, min_dist


def main() -> None:
    # Файл для логов
    out_path = Path("angle_debug_log.csv")

    # Загружаем centerline (тот же, что использует env.TrackmaniaEnv)
    centerline_path = Path(cfg.env.centerline_path)
    center = track.CenterLine(centerline_path)

    # Конфиг TMIClient — берём те же значения, что и в config.py,
    # НО prevent_finish ставим в False, чтобы игра могла нормально завершить заезд.
    tmi_cfg = TMIConfig(
        host=cfg.tmiface.host,
        port=cfg.tmiface.port,
        game_speed=cfg.tmiface.game_speed,
        prevent_finish=False,
        connect_timeout_s=cfg.tmiface.connect_timeout,
        server_name=cfg.tmiface.server_name,
    )
    client = TMIClient(cfg=tmi_cfg)

    print(f"[debug] Connecting to TMInterface plugin on {tmi_cfg.host}:{tmi_cfg.port}...")
    client.start()
    print("[debug] Connected.")

    # Ждём первый state
    if not client.wait_for_state(timeout=5.0):
        client.close()
        raise RuntimeError("TMInterface не прислал первое состояние (timeout)")

    print(f"[debug] Centerline loaded from: {centerline_path.resolve()}")
    print(f"[debug] Track length (s_max): {center.length:.3f}")
    print(f"[debug] Logging to: {out_path.resolve()}")

    tick = 0

    # Для вычисления heading по смещению в XZ, как в env.py
    last_pos: Tuple[float, float, float] | None = None
    last_heading: float = 0.0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "tick",
                "race_time",
                "cp_index",
                "lap",
                "pos_x",
                "pos_y",
                "pos_z",
                "yaw_rad",
                "yaw_deg",
                "heading_rad",
                "heading_deg",
                "speed",
                "s",
                "d",
                "tangent_rad",
                "tangent_deg",
                "ang_diff_yaw_rad",
                "ang_diff_yaw_deg",
                "ang_diff_heading_rad",
                "ang_diff_heading_deg",
                "seg_idx",
                "seg_t",
                "dist_to_seg",
                "dist_cp_dummy",
                "wall_contact_count",
            ]
        )

        while True:
            # ждём следующий тик
            if not client.wait_for_state(timeout=1.0):
                print("[debug] timeout ожидания состояния — прекращаю логирование")
                break

            state = client.get_state()
            if not state:
                continue

            pos_x, pos_y, pos_z = state["position"]
            x, y, z = pos_x, pos_y, pos_z

            yaw = float(state.get("yaw", 0.0))
            speed = float(state.get("speed", 0.0))
            race_time = float(state.get("race_time", 0.0))
            cp_index = int(state.get("cp_index", -1))
            lap = int(state.get("lap", 0))
            wall_count = int(state.get("wall_contact_count", 0))

            # --- heading по смещению в XZ (как в env._state_to_obs) ---
            heading = float(last_heading)
            if last_pos is not None:
                dx = x - last_pos[0]
                dz = z - last_pos[2]
                if dx * dx + dz * dz > 1e-6:
                    heading = math.atan2(dz, dx)

            last_pos = (x, y, z)
            last_heading = heading

            # Геометрия относительно центра
            s, d, tangent_angle, seg_idx, seg_t, dist_seg = project_with_debug(
                center, (pos_x, pos_y, pos_z)
            )

            # Угол yaw относительно касательной (старый вариант)
            ang_yaw = angle_diff(yaw, tangent_angle)
            # Угол heading относительно касательной (новый, как в env)
            ang_heading = angle_diff(heading, tangent_angle)

            # Для полноты — расстояние до "следующего cp" как до конца круга
            dist_cp_dummy = max(center.length - s, 0.0)

            writer.writerow(
                [
                    tick,
                    f"{race_time:.3f}",
                    cp_index,
                    lap,
                    f"{pos_x:.3f}",
                    f"{pos_y:.3f}",
                    f"{pos_z:.3f}",
                    f"{yaw:.6f}",
                    f"{math.degrees(yaw):.3f}",
                    f"{heading:.6f}",
                    f"{math.degrees(heading):.3f}",
                    f"{speed:.3f}",
                    f"{s:.3f}",
                    f"{d:.3f}",
                    f"{tangent_angle:.6f}",
                    f"{math.degrees(tangent_angle):.3f}",
                    f"{ang_yaw:.6f}",
                    f"{math.degrees(ang_yaw):.3f}",
                    f"{ang_heading:.6f}",
                    f"{math.degrees(ang_heading):.3f}",
                    seg_idx,
                    f"{seg_t:.3f}",
                    f"{dist_seg:.3f}",
                    f"{dist_cp_dummy:.3f}",
                    wall_count,
                ]
            )

            tick += 1

            if bool(state.get("finished", False)):
                print("[debug] finished == True, выхожу из цикла логирования")
                break

    client.close()
    print("[debug] Соединение закрыто, лог готов.")


if __name__ == "__main__":
    main()
