"""AI-assisted robotic teleoperation for precision peg-in-hole insertion.

Top-level package. See submodules:

- domain   : interfaces (Protocols) and core dataclasses (Observation, Command, ...).
- sim      : MuJoCo scene wrapper, sensor reading, rendering.
- control  : backbone controller (operational-space IK, impedance) — the always-on substrate.
- input    : input strategies (vision, keyboard, scripted noisy-human) behind a common interface.
- expert   : analytical privileged-info expert (data-generation supervisor).
- policy   : residual correction policy (BC-trained neural network).
- data     : data-generation pipeline, dataset loaders.
- eval     : evaluation harness (passive observer, KPI computation, ablation orchestration).
- common   : shared utilities (logging, types, math).
"""

__version__ = "0.0.1"
