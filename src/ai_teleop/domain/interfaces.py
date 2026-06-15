"""Assistance seam — Protocol definitions for command-source and correction layers.

Both protocols are @runtime_checkable so conformance can be asserted in tests.
mypy structural typing is the correctness guarantee; isinstance checks are for
test clarity only.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation

from .delta import Delta


@runtime_checkable
class InputStrategy(Protocol):
    """Produces the base per-tick EE-pose Command from an Observation."""

    def get_command(self, observation: Observation) -> Command: ...


@runtime_checkable
class AssistProvider(Protocol):
    """Returns a correction Delta to add on top of the input's base Command."""

    def get_delta(self, observation: Observation, command: Command) -> Delta: ...
