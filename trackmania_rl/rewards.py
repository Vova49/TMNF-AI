# rewards.py
"""Reward functions for TrackMania RL project (TMNF + TMInterface).

Ключевые принципы под быстрейший проезд круга:
- Вознаграждаем *выравненный прогресс* (вперёд по трассе и по направлению трассы).
- НЕ штрафуем за отъезд от центра (рейсинг-лайн допускает смещения).
- Жёстко наказываем касания стен.
- Бонусы за чекпоинты + shaping по приближению к следующему CP.
- Доп. санкции за движение назад/простой + лёгкий штраф за излишнюю дёрганность (по углу).

Ожидаемые поля в `state` (dict):
    s: float                  # прогресс вдоль центра (м)
    ang_diff: float           # yaw - tangent_angle в радианах [-pi, pi]
    speed: float              # м/с
    race_time: int | float    # монотонное время
    cp_index: int             # текущий чекпоинт (или следующий к взятию)
    has_any_lateral_contact: bool  # из SceneVehicleCar.has_any_lateral_contact

Опциональные поля, если есть — будут учтены:
    dist_cp: float            # расстояние до след. чекпоинта вдоль s (м)
    curvature: float          # кривизна центра линии в текущей точке (1/м), |k| больше в крутых поворотах
    just_respawned: bool      # true если был ресет/телепорт в этот тик
"""
from __future__ import annotations

from math import cos, pi
from typing import Dict, Any, Optional

try:
    from trackmania_rl import config as cfg
except Exception:
    cfg = None


def _safe(state: Optional[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    if not state:
        return default
    v = state.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


def _bool(state: Optional[Dict[str, Any]], key: str) -> bool:
    if not state:
        return False
    return bool(state.get(key, False))


def _cp_index(state: Optional[Dict[str, Any]]) -> int:
    if not state:
        return 0
    try:
        return int(state.get("cp_index", 0))
    except Exception:
        return 0


def compute_reward(
    prev_state: Optional[Dict[str, Any]],
    cur_state: Optional[Dict[str, Any]],
    info: Optional[Dict[str, Any]] = None,
) -> float:
    """Подсчёт награды за один RL-шаг без штрафа за удаление от центра трассы."""
    if prev_state is None or cur_state is None:
        return 0.0

    R = getattr(cfg, "reward", None) if cfg is not None else None
    W_PROGRESS = float(getattr(R, "W_PROGRESS", 1.3))
    W_CP = float(getattr(R, "W_CP", 10.0))
    W_CP_SHAP = float(getattr(R, "W_CP_SHAP", 0.02))
    W_WALL = float(getattr(R, "W_WALL", 4.0))
    W_IDLE = float(getattr(R, "W_IDLE", 0.1))
    W_BACKWARD = float(getattr(R, "W_BACKWARD", 2.0))
    W_SMOOTH_ANG = float(getattr(R, "W_SMOOTH_ANG", 0.03))
    W_FALL = float(getattr(R, "W_FALL", 20.0))
    BACKWARD_THRESH = float(getattr(R, "BACKWARD_THRESH", 0.25))
    ALIGN_GAMMA = float(getattr(R, "ALIGN_GAMMA", 1.0))
    CURV_BETA = float(getattr(R, "CURV_BETA", 0.0))
    IDLE_SPEED_THRESH = float(getattr(R, "IDLE_SPEED_THRESH", 15.0))

    s_prev, s_cur = _safe(prev_state, "s"), _safe(cur_state, "s")
    ds = s_cur - s_prev

    # Респавн/телепорт — не считаем отрицательный ds
    just_respawned = _bool(cur_state, "just_respawned") or (
        _safe(cur_state, "race_time", 0.0) < _safe(prev_state, "race_time", 0.0)
    )
    if just_respawned and ds < 0:
        ds = 0.0

    ang_prev, ang_cur = _safe(prev_state, "ang_diff"), _safe(cur_state, "ang_diff")
    dang = abs(ang_cur - ang_prev)

    # 1) Выравненный прогресс: ds * max(0, cos(ang))^gamma
    # - если едем “поперёк/назад” по направлению (cos<0) — вклад 0
    # - это не штрафует смещения от центра, только ориентирование vs направление трассы
    align = cos(ang_cur)
    if align < 0.0:
        align_eff = 0.0
    else:
        align_eff = align ** ALIGN_GAMMA

    # Кривизна (если есть) — нормируем прогресс: в крутых поворотах «разрешаем» меньший темп
    kappa = abs(_safe(cur_state, "curvature", 0.0))
    curv_div = (1.0 + CURV_BETA * kappa) if CURV_BETA > 0.0 else 1.0

    r_progress = W_PROGRESS * max(0.0, ds) * align_eff / curv_div

    # 2) Бонус за чекпоинт
    cp_prev, cp_cur = _cp_index(prev_state), _cp_index(cur_state)
    took_cp = cp_cur > cp_prev
    r_cp = W_CP if took_cp else 0.0

    # 3) Shaping по приближению к CP, если dist_cp есть
    dist_prev = _safe(prev_state, "dist_cp", float("nan"))
    dist_cur = _safe(cur_state, "dist_cp", float("nan"))
    r_cp_shap = 0.0
    if dist_prev == dist_prev and dist_cur == dist_cur:
        r_cp_shap = W_CP_SHAP * (dist_prev - dist_cur)

    # 4) Касание стены — штраф КАЖДЫЙ тик контакта
    contact_now = bool(cur_state.get("has_any_lateral_contact", False))
    r_wall = -W_WALL if contact_now else 0.0

    # 5) Штраф за простой
    speed_cur = _safe(cur_state, "speed", 0.0)
    r_idle = -W_IDLE if speed_cur < IDLE_SPEED_THRESH else 0.0

    # 6) Штраф за движение назад по s (не респавн)
    r_backward = 0.0
    if ds < -BACKWARD_THRESH and not just_respawned:
        r_backward = -W_BACKWARD * abs(ds)

    # 7) Сглаживание траектории: наказываем резкие изменения угла относительно тангенса
    r_smooth = -W_SMOOTH_ANG * dang / pi  # нормируем к [0,1]

    # 8) Крупный штраф за падение ниже минимальной высоты Y
    r_fall = 0.0
    try:
        pos = cur_state.get("position")
        if pos is not None and len(pos) >= 2:
            y = float(pos[1])
            death_y = 15.0
            if cfg is not None:
                try:
                    death_y = float(
                        getattr(getattr(cfg, "env", object()), "death_y_threshold", 15.0)
                    )
                except Exception:
                    death_y = 15.0
            if y < death_y:
                r_fall = -W_FALL
    except Exception:
        r_fall = 0.0

    total = (
        r_progress
        + r_cp
        + r_cp_shap
        + r_wall
        + r_idle
        + r_backward
        + r_smooth
        + r_fall
    )

    if info is not None:
        info.update(
            {
                "r_progress": r_progress,
                "r_cp": r_cp,
                "r_cp_shap": r_cp_shap,
                "r_wall": r_wall,
                "r_idle": r_idle,
                "r_backward": r_backward,
                "r_smooth": r_smooth,
                "r_fall": r_fall,
                "fell_below_track": bool(r_fall < 0.0),
                "r_total": total,
                "ds": ds,
                "align": align,
                "align_eff": align_eff,
                "kappa": kappa,
                "curv_div": curv_div,
                "dang": dang,
                "cp_prev": cp_prev,
                "cp_cur": cp_cur,
                "wall_contact": bool(
                    cur_state.get("has_any_lateral_contact", False)
                ),
            }
        )
    return float(total)
