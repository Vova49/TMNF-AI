"""Backward-compatible shim to the new TMInterface plugin client."""
from trackmania_rl.tmi_interaction.client import TMIClient, TMIConfig

__all__ = ["TMIClient", "TMIConfig"]
