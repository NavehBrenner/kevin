"""Small pure geometry helpers shared across layers.

Leaf of the dependency DAG (must not import other ai_teleop subpackages). The
only dependency beyond numpy is MuJoCo, used here purely as a math library.
"""

from __future__ import annotations

import mujoco
import numpy as np


def mat3_to_quat(mat_flat: np.ndarray) -> np.ndarray:
    """Convert a flattened 3x3 row-major rotation matrix to a (w,x,y,z) unit quat.

    `data.site_xmat` / `data.xmat` are stored as 9-element row-major flat arrays,
    which is exactly the layout MuJoCo's ``mju_mat2Quat`` expects.
    """
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(mat_flat).reshape(9))
    return quat
