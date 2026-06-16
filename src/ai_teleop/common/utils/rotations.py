import mujoco
import numpy as np


def quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def axis_from_quat(quaternion: np.ndarray, axis: int) -> np.ndarray:
    rotation = quat_to_matrix(quaternion)
    return rotation[:, axis]


def quat_to_6d(quaternion: np.ndarray) -> np.ndarray:
    matrix = quat_to_matrix(quaternion)

    return np.concatenate([matrix[:, 0], matrix[:, 1]])
