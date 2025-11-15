"""Global configuration for the Trackmania RL project.

This module centralises all tunable hyper-parameters and runtime settings so
that other components (environment, reward computation, client wrapper, etc.)
can import them instead of hard-coding values.

Feel free to tweak the numbers or even load them from ``.toml`` / ``.yaml`` in
future – as long as you keep the public attribute names intact the rest of the
code will continue to work.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Dataclass groups – keep related parameters together for clarity
# ---------------------------------------------------------------------------
@dataclass
class TMInterfaceCfg:
    """Settings that affect low-level connection to TMInterface."""

    server_name: str = "TMInterface0"  # Windows named pipe / TCP server
    connect_timeout: float = 5.0  # seconds to wait until giving up
    game_speed: float = 1.0  # 1.0 – realtime, >1 faster, <1 slower
    prevent_finish: bool = True  # call iface.prevent_simulation_finish()


@dataclass
class EnvCfg:
    """Gym-style environment parameters."""

    # How many *physics ticks* we aggregate into a single RL step
    ticks_per_step: int = 2
    # Maximum number of physics ticks in a single episode (safety cut-off)
    episode_max_ticks: int = 10_000
    # Default path to the saved centre-line (can be overridden at runtime)
    centerline_path: str = "coords.txt"

    respawn_cooldown_ticks: int = 20


@dataclass
class RewardCfg:
    """Coefficients for shaping the reward signal."""

    # simple speed based shaping term (reward = scale * speed)
    forward_speed_scale: float = 1e-3


# ---------------------------------------------------------------------------
# Instantiate *mutable* objects so the values can be patched at runtime
# ---------------------------------------------------------------------------
tmiface = TMInterfaceCfg()
env = EnvCfg()
reward = RewardCfg()

# ---------------------------------------------------------------------------
# Convenience top-level aliases (read-only!) – importers can do::
#     from trackmania_rl import config as cfg
#     cfg.TICKS_PER_STEP
# instead of deep attribute chains.
# ---------------------------------------------------------------------------
TICKS_PER_STEP: int = env.ticks_per_step
EPISODE_MAX_TICKS: int = env.episode_max_ticks
FORWARD_SPEED_SCALE: float = reward.forward_speed_scale
SERVER_NAME: str = tmiface.server_name
