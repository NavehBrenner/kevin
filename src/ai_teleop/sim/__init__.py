"""MuJoCo simulation wrapper.

Owns the MJCF model, the physics state, the sensor reads, and the rendering paths
(interactive viewer + headless wrist-camera).
"""

from ai_teleop.sim.scene import SimEnv

__all__ = ["SimEnv"]
