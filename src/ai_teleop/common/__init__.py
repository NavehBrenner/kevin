"""Shared utilities: logging, common math, type aliases. helper functions

Should never import from other ai_teleop subpackages — this is the leaf of the
dependency DAG.
"""

from ai_teleop.common.command import Command
from ai_teleop.common.geometry import (
    axis_from_quat,
    mat3_to_quat,
    quat_conjugate,
    quat_mul,
    quat_to_6d,
    quat_to_matrix,
)
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
    "axis_from_quat",
    "configure_from_args",
    "configure_logging",
    "get_logger",
    "mat3_to_quat",
    "quat_conjugate",
    "quat_mul",
    "quat_to_6d",
    "quat_to_matrix",
]
