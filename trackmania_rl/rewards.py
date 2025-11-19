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

# ---------------------------------------------------------------------------
# Конфиг (значения можно переопределить через config.reward.*)
# ---------------------------------------------------------------------------
try:
    from trackmania_rl import config as cfg

    _R = getattr(cfg, "reward", cfg)
except Exception:
    cfg = None

    class _R:
        pass


def _get(name: str, default: float) -> float:
    return float(getattr(_R, name, default))


# Веса
W_PROGRESS = _get("W_PROGRESS", 1.3)  # выравненный прогресс
W_CP = _get("W_CP", 10.0)  # бонус за чекпоинт
W_CP_SHAP = _get("W_CP_SHAP", 0.02)  # shaping по dist_cp
W_WALL = _get("W_WALL", 6.0)  # штраф за касание стены
W_IDLE = _get("W_IDLE", 0.002)  # штраф за простой
W_BACKWARD = _get("W_BACKWARD", 2.0)  # штраф за явное движение назад
W_SMOOTH_ANG = _get("W_SMOOTH_ANG", 0.03)  # мягкое наказание за |Δang_diff|
W_FALL = _get("W_FALL", 80.0)  # крупный штраф за падение ниже трека (Y < death_y)

# Параметры формы
BACKWARD_THRESH = _get("BACKWARD_THRESH", 0.25)  # м за шаг, ниже считаем шумом
ALIGN_GAMMA = _get("ALIGN_GAMMA", 1.0)  # степень для cos(ang) в прогрессе
CURV_BETA = _get("CURV_BETA", 0.0)  # если >0, прогресс делится на (1+β|k|) — не наказывает за смещения на рейсинг-лайне


# Новый штраф за "залипание" в стену при зажатом газе
W_STUCK = _get("W_STUCK", 4.0)  # можно потом подкрутить

# Пороговые параметры для детекции залипания
STUCK_SPEED_MAX = _get("STUCK_SPEED_MAX", 4.0)          # м/с, считаем что почти стоим
STUCK_MIN_PREV_SPEED = _get("STUCK_MIN_PREV_SPEED", 5.0)  # до этого ехали хотя бы столько
STUCK_DS_MAX = _get("STUCK_DS_MAX", 0.10)               # прогресс по s почти нулевой
STUCK_GAS_MIN = _get("STUCK_GAS_MIN", 0.5)              # газ нажимаем "по-настоящему"

# Максимум для счётчика залипания — защищает от слишком больших отрицательных наград,
# если агент долго торчит у стены. Можно переопределить через config.reward.MAX_STUCK_TICKS
MAX_STUCK_TICKS = int(_get("MAX_STUCK_TICKS", 25))


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
    wall_lateral = bool(cur_state.get("has_any_lateral_contact", False))
    r_wall = -W_WALL if wall_lateral else 0.0

    # 5) "Залипание" в стену/препятствие:
    # жмём газ, но почти не двигаемся вперёд и скорость очень маленькая.
    gas_cmd = _safe(cur_state, "cmd_gas", 0.0)
    speed_prev = _safe(prev_state, "speed", 0.0)
    speed_cur = _safe(cur_state, "speed", 0.0)
    stuck_prev = int(_safe(prev_state, "stuck_ticks", 0.0))

    base_blocked = (
            gas_cmd > STUCK_GAS_MIN  # реально жмём газ
            and not just_respawned  # не в момент респавна
            and speed_cur < STUCK_SPEED_MAX  # почти стоим
            and abs(ds) < STUCK_DS_MAX  # прогресс по s ≈ 0
    )

    # Запускаем "счётчик залипания" либо с момента сильного торможения,
    # либо продолжаем его, если уже были в таком состоянии.
    stuck_now = base_blocked and (
            speed_prev > STUCK_MIN_PREV_SPEED or stuck_prev > 0
    )
    stuck_ticks = stuck_prev + 1 if stuck_now else 0

    # Ограничиваем рост счётчика, чтобы штраф не рос бесконечно
    stuck_ticks = min(stuck_ticks, MAX_STUCK_TICKS)

    if isinstance(cur_state, dict):
        # сохраняем счётчик в стейте, чтобы на следующем шаге его видеть как prev_state["stuck_ticks"]
        cur_state["stuck_ticks"] = float(stuck_ticks)

    r_stuck = -W_STUCK * float(stuck_ticks) if stuck_now else 0.0

    # 6) Штраф за простой (мягкий базовый штраф)
    r_idle = -W_IDLE

    # 7) Штраф за движение назад по s (не респавн)
    r_backward = 0.0
    if ds < -BACKWARD_THRESH and not just_respawned:
        r_backward = -W_BACKWARD * abs(ds)

        # 8) Сглаживание траектории...
    r_smooth = -W_SMOOTH_ANG * dang / pi

    # 9) Крупный штраф за падение ниже минимальной высоты Y
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
        + r_stuck
        + r_idle
        + r_backward
        + r_smooth
        + r_fall
    )

    if info is not None:
        wall_front_like = bool(stuck_now)
        info.update(
            {
                "r_progress": r_progress,
                "r_cp": r_cp,
                "r_cp_shap": r_cp_shap,
                "r_wall": r_wall,
                "r_stuck": r_stuck,
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
                "wall_contact": wall_lateral or wall_front_like,
                "wall_lateral": wall_lateral,
                "wall_front_like": wall_front_like,
                "stuck_ticks": stuck_ticks,
                "stuck_base_blocked": base_blocked,
            }
        )
    return float(total)
