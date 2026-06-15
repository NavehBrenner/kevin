"""Analytical privileged-info expert.

Closed-form supervisor that consumes the full simulation state plus the
noisy-human's current command and emits the corrective delta a residual policy
should output. Used at data-generation time only; never deployed.

See ``docs/design/expert-corrections.md`` and ``docs/milestone-4-spec.md``.
"""

from ai_teleop.expert.expert import Expert

__all__ = ["Expert"]
