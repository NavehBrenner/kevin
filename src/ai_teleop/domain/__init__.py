"""Domain layer: interfaces (Protocols) and core dataclasses.

All concrete implementations elsewhere in the package depend on the abstractions
defined here, not the other way around (Dependency Inversion).

No imports from ai_teleop.sim — Command/Observation remain sim-independent.
"""

from ai_teleop.domain.delta import ZERO_DELTA, Delta, NoAssist, apply_delta, clamp_delta
from ai_teleop.domain.interfaces import AssistProvider, InputStrategy

__all__ = [
    "Delta",
    "ZERO_DELTA",
    "clamp_delta",
    "apply_delta",
    "NoAssist",
    "InputStrategy",
    "AssistProvider",
]
