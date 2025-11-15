"""Trackmania RL package.

Core modules live directly in this package.  Import helpers keep them
accessible via ``import trackmania_rl.<module>``.
"""
from importlib import import_module

__all__ = [
    "config",
    "env",
    "policy",
    "rewards",
    "tmi_client",
    "track_pipeline",
]

for _name in list(__all__):
    try:
        globals()[_name] = import_module(f"{__name__}.{_name}")
    except ModuleNotFoundError:
        # Allow optional modules to be missing during early development
        pass
