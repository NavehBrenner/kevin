# AI-Assisted Robotic Teleoperation for Precision Insertion

A simulated robotic arm performs peg-in-hole insertions under shared-autonomy control: a human operator provides coarse 6-DoF commands (webcam-tracked hand motion), and a vision-conditioned residual policy issues real-time micro-corrections so insertions reliably succeed even when the human's input is noisy.

Built in [MuJoCo](https://mujoco.org) for the Franka Emika Panda. The residual policy is trained via behavioral cloning against a scripted privileged-info expert. Two configurations — human-only and human + learned residual — are compared head-to-head in a KPI ablation, on a shared always-on impedance backbone.

> **Status**: pre-implementation. Repository is currently being scaffolded. See [project-scope.md](./project-scope.md) for the full project definition and [docs/milestone-1-spec.md](./docs/milestone-1-spec.md) for the current work item.

## Project context

Course project for *Workshop in Autonomous Systems Simulation* (OpenU course 20973, fall 2026). Solo project. Final submission deadline: 2026-08-31.

## Documents

- **[project-scope.md](./project-scope.md)** — full scope, design decisions, KPIs, architecture overview, deferred design questions.
- **[docs/milestone-1-spec.md](./docs/milestone-1-spec.md)** — current milestone: simulation environment online.

## Quick start

Not yet — implementation begins at Milestone 1. Once M1 lands, this section will document setup and the smoke test.

## License

To be added (likely MIT).
