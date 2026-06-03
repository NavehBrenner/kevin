# Acronym & Abbreviation Dictionary

Approved short forms for this codebase. Any abbreviation used in code **must** appear here.
Add a row when you introduce a new short form; expand the "full form" and note the reason.

| Short form | Full form | Reason / notes |
|---|---|---|
| `env` | environment | Very common in RL/sim APIs; verbose form is long in compound names |
| `obs` | observation | Standard RL term; appears in function signatures throughout |
| `cfg` | config | Configuration objects are passed around frequently |
| `q` | joint angles (generalized coordinates) | Standard robotics / MuJoCo notation |
| `R` | rotation matrix | Standard linear algebra notation |
| `T` | homogeneous transform matrix | Standard robotics notation |
| `dt` | time step (delta time) | Universal physics/simulation convention |
| `sim` | simulation | Domain prefix used in class names (e.g. `SimEnv`) |
| `i`, `j`, `k` | loop indices | Universal; fine in tight loops |
| `n` | count / number-of | Only when the referent is clear from context (e.g. `n_joints`) |
| `pos` | position | Acceptable in compound attribute names (e.g. `eef_pos`) |
| `vel` | velocity | Acceptable in compound attribute names (e.g. `eef_vel`) |
| `eef` | end-effector | Robotics term; long in compound names |
| `BC` | behavioral cloning | ML method abbreviation used in comments / class names |
| `DoF` | degrees of freedom | Standard robotics / engineering abbreviation |
