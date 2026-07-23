"""Small pure geometry helpers shared across layers — the one rotation toolbox.

Quaternion layout is ``(w, x, y, z)`` everywhere, matching MuJoCo and
``Observation.ee_pose`` / ``Command.target_quaternion``.

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


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ∘ b`` of two (w,x,y,z) quaternions (rotate by b then a)."""
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, a, b)
    return result


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse, for a unit quat) of a (w,x,y,z) quaternion."""
    result = np.zeros(4)
    mujoco.mju_negQuat(result, q)
    return result


def quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """(w,x,y,z) unit quaternion → its 3x3 rotation matrix."""
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def axis_from_quat(quaternion: np.ndarray, axis: int) -> np.ndarray:
    """The rotated basis vector ``axis`` (0=x, 1=y, 2=z) as a world-frame unit vector.

    The idiom behind every "which way is this body pointing" question in the stack:
    the peg's long axis is its local +z (``axis=2``), a hole's bore is its local +x
    (``axis=0``).
    """
    return quat_to_matrix(quaternion)[:, axis]


def quat_to_6d(quaternion: np.ndarray) -> np.ndarray:
    """(w,x,y,z) quaternion → the 6D continuous rotation representation.

    The first two columns of the rotation matrix. Quaternions are discontinuous as
    a network input/output (antipodal double cover, ``q`` and ``−q`` are the same
    rotation); this 6D form is continuous, which is why the policy's command and
    proprioception streams carry orientation this way. See
    ``project-wiki/concepts/6d-rotation-representation.md``.
    """
    matrix = quat_to_matrix(quaternion)
    return np.concatenate([matrix[:, 0], matrix[:, 1]])
