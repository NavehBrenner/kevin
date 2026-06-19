"""Shared utilities: logging, common math, type aliases. helper functions

Should never import from other ai_teleop subpackages — this is the leaf of the
dependency DAG.
"""

from ai_teleop.common import utils
from ai_teleop.common.command import Command
from ai_teleop.common.log import (
    add_logging_arguments,
    configure_from_args,
    configure_logging,
    get_logger,
)
from ai_teleop.common.observation import Observation
from ai_teleop.common.seating import PEG_HALF_LENGTH, SeatingGeometry

__all__ = [
    "PEG_HALF_LENGTH",
    "Command",
    "Observation",
    "SeatingGeometry",
    "add_logging_arguments",
    "configure_from_args",
    "configure_logging",
    "get_logger",
    "utils",
]
