# scripts/

Executable entry points for the project workflow.

Planned scripts (populated across milestones):

- `smoke_test_sim.py`   — M1: load scene, dump sensor readings + wrist-cam PNG.
- `manual_drive.py`     — M2/M3: keyboard-drive the arm for visual debugging.
- `generate_data.py`    — M4: run unattended data-generation episodes.
- `train_policy.py`     — post-M4: BC training on logged data.
- `run_eval.py`         — post-M4: ablation orchestration + KPI computation.
- `run_demo.py`         — final: live webcam-driven shared-autonomy demo.
