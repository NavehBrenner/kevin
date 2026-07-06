# Expert Corrections — The Analytical Privileged-Info Expert

Companion docs: [problem-structure.md](problem-structure.md) (notation, ground truth) · [human-generation.md](human-generation.md) · [policy-model.md](policy-model.md). Refines [`../../project-scope.md`](../../project-scope.md) Component 3.

This document specifies the **expert**: the closed-form, geometry-driven controller that generates the supervised target `Δ*_t` for behavioral cloning. The expert is a *tool*, not a research contribution — the contribution is the deployed policy reproducing its output *without* privileged state. The expert is allowed to "cheat" (it reads true poses); the policy is not.

## Contract

- **Inputs**: privileged true state `s_t` (true peg-tip pose, true target-hole pose `→ p_hole, n`, full arm state) **+** the noisy operator command `c_t` for this step.
- **Output**: a clamped correction `Δ*_t = (Δposition ∈ ℝ³, Δorientation ∈ ℝ³ axis-angle, Δgrip ∈ ℝ¹)`, the **same signature** as the policy's output. This symmetry is what makes BC exact: the policy mimics `Δ*_t` directly (see [policy-model.md](policy-model.md)).
- **Type**: analytical state-feedback law. No RL, no learning, no human recordings. Deterministic given `(s_t, c_t)`.

The expert computes a **desired pose** from geometry, expresses the gap between that desired pose and the operator's command as a correction, then gates and clamps it. It is *not* a noise canceller — it never sees the injected noise (see [human-generation.md](human-generation.md)); it works purely from where the peg/hole actually are.

## Geometry, per step

Let the true peg tip be `p_tip`, peg long-axis unit vector `a`; target hole entry `p_hole`, insertion-axis unit vector `n` (points into the hole). Define the tip→hole error and split it along/around the insertion axis:

```
e      = p_hole − p_tip                 # 3D position error (tip to hole entry)
e_ax   = (e · n) · n                    # component ALONG the insertion axis
e_lat  = e − e_ax                       # component LATERAL to the axis (the misalignment that must vanish first)
d      = ‖e‖                            # scalar approach distance  (also used by the gate g below)
```

### Phased desired motion

The expert enforces the natural peg-in-hole order — **align laterally and angularly first, then advance** — because pushing in while misaligned just jams the peg on the rim.

1. **Lateral alignment.** Drive `e_lat → 0`: desired lateral position shift `= e_lat` (move the tip onto the hole axis line).
2. **Angular alignment.** Compute `R_align` so the peg axis `a` maps onto the hole axis `n` (smallest rotation taking `a` to `n`; off-axis pitch/roll only — yaw is irrelevant for a round peg). Desired orientation `R_des = R_align · R_cmd`.
3. **Axial advance — gated by alignment.** Advance along `−n` (into the hole) at a capped speed, but **only when lateral and angular error are within tolerance** (`‖e_lat‖ < ε_lat` and angular error `< ε_ang`). Until then, axial advance is suppressed so the peg doesn't drive into the rim. The chamfer + lateral compliance of the backbone handle the final rim-guided seating physically; the expert just keeps the peg aligned and feeds it in.
4. **Approach-speed braking (LAB-98).** Under the deployment controller config (`joint_damping=1.5`, the config the corpus is generated under since LAB-96) the arm tracks the operator's command tightly, so a hasty episode meets the wall at its drawn sweep speed and trips the 30 N watchdog — a failure mode the kd=4 data-gen controller used to suppress *inside the controller* (which is why the pre-LAB-98 expert never needed this term). The expert therefore also governs the command's **axial lead** — how far past the arm the command sits along the bore:

   ```
   lead      = (c_t.position − p_ee) · n          # command's lead ahead of the arm, along the bore
   allowed   = brake_gain · d + brake_lead_floor  # distance-proportional allowance
   Δ_brake   = −max(0, lead − allowed) · n
   ```

   The effective target the impedance law chases becomes a controlled "carrot" at most `allowed` ahead of the arm, so approach speed (∝ lead under an impedance law) decays with distance — deceleration before contact, the "assist stops you slamming the wall" behavior. Two properties worth noting: the brake reads only `c_t − p_ee` — **non-privileged** streams, so unlike the aim correction this component is fully inferable by the deployed policy; and its authority is structurally bounded by the shared Δ-clamp, so operator sweeps faster than the clamp can absorb still crash (the honest-ceiling residual, measured in the LAB-98 sweep at ±2 cm; LAB-100 widened the bound to ±3 cm — the smallest that stops the brake saturating on success episodes — converting part of that residual). `brake_gain = 0` disables the term (the pre-LAB-98 aim-only expert, bit-exact).

5. **Grip.** Hold the baseline grip force by default; **reduce** grip on a detected jam signature (so a slightly-angled peg can slip into alignment — the grip-modulation behavior the scope wants the policy to learn). Increase back to baseline once seated.

This yields a **desired pose** `pose_des = (p_tip + desired position shift, R_des)` and a desired grip.

## From desired pose to correction `Δ*`

The expert outputs a **residual on the operator's command**, not an absolute pose. So it differences the desired pose against the operator command `c_t`:

```
Δ_full.position    = pose_des.position − c_t.position
Δ_full.orientation = log( R_des · R_cmdᵀ )           # axis-angle, the rotation taking c_t's orientation to desired
Δ_full.grip        = grip_des − c_t.grip             # (baseline-relative grip rule)
```

i.e. `Δ_full = pose_des ⊖ c_t` in the pose-difference notation of [problem-structure.md](problem-structure.md). Adding `Δ_full` to `c_t` would command exactly `pose_des`.

### The distance gate `g(d)` — far-field zero by construction

We do **not** emit `Δ_full` raw. We multiply by a smooth distance gate so the correction is **zero when far from the hole** and ramps to full authority near contact:

```
Δ*_t = clamp(  g(d) · Δ_full  )

g(d) = 0                       for d ≥ d_far
     = smoothstep(d_far, d_near, d)   for d_near < d < d_far   # smooth 0→1 ramp
     = 1                       for d ≤ d_near
```

(`smoothstep` is the standard Hermite `3x²−2x³` blend, so `g` is C¹ — no jerk at the boundaries.) Rationale:

- **Far from the hole there is genuinely nothing to correct**: the operator's coarse approach is fine at the centimeter scale, the meaningful work is all in the last millimeters, and — critically — the deployed policy has *no reliable signal* far away anyway (F/T is ~0 in free space; the camera only resolves the hole on approach). Teaching a near-zero far-field correction matches what the policy can actually support and keeps the assist from fighting the operator during transit.
- Because `g(d_far)=0` **exactly and by construction**, the expert's far-field correction is *structurally* zero, not approximately zero. We do not rely on the geometry happening to cancel; we hard-gate it.

The clamp (`±3 cm / ±10° / ±5 N` per step; the position bound was ±2 cm before LAB-100 widened it to give the brake authority over the fast operator tail) is applied last, as the final safety bound — identical bounds to the deployed policy.

## Validating "Δ ≈ 0 far from the wall"

Four independent checks, by construction → in the data → in the trained policy → in deployment:

1. **By construction (unit test).** Assert `g(d) == 0` for `d ≥ d_far`, hence `Δ* == 0`, across a grid of far-field states. This is a deterministic property of the gate — a cheap pytest that can never silently regress.
2. **In the dataset (scatter).** Plot `‖Δ*_position‖` (and angular/grip magnitudes) vs `d` over a sample of logged episodes. Expect a clean floor at zero for `d ≥ d_far`, ramping up only near contact. Catches mis-set `d_far` or a geometry bug.
3. **In the trained policy (scatter).** Same plot for `‖π_θ(o_t)‖` vs `d` on held-out episodes. The *policy* should have learned the near-zero far-field behavior from the data. Divergence here flags covariate-shift / under-training, not an expert bug.
4. **In deployment (free-space no-op test).** Run the policy on a trajectory that stays in free space (never approaches a hole) and assert the integrated correction stays within a small tolerance band — the assist must be a near-no-op when there's nothing to do.

Checks 1–2 validate the *expert/label*; 3–4 validate the *policy*. They are listed together here because "far-field correction is zero" is a property we want to hold end-to-end, and each stage has its own owner.

## Why an analytical expert (not RL / not human)

- **Free, perfect, abundant supervision**: it runs at sim speed, unattended, with privileged ground truth — so every step is labeled with the geometrically-correct correction. No reward shaping, no human teleop sessions.
- **Standard for peg-in-hole BC**: closed-form geometric experts are the canonical demonstrator for this task family.
- **Honest difficulty split**: the expert gets to cheat (true poses); the policy must match it from partial observation. That asymmetry *is* the project. If we gave the policy privileged state too, there'd be no learning problem.

The expert's one job is to be *correct given the state*, even on failing trajectories — which is why we keep failure episodes in the data (state coverage), per the scope.

## Open / to-calibrate

- `d_near`, `d_far` for the gate; `ε_lat`, `ε_ang` alignment tolerances; axial speed cap — calibrated alongside the contact/chamfer tuning.
- Jam-detection signature for grip reduction (F/T pattern threshold) — tuned against logged jam episodes.
- Whether to inject small noise into the expert's action during data-gen to demonstrate *recovery* (covariate-shift mitigation — see [problem-structure.md](problem-structure.md)); decided empirically if open-loop BC underperforms.

---

### Provenance note

Records the analytical expert design from the 2026-05/06 scoping discussions, refining [`../../project-scope.md`](../../project-scope.md) Component 3. Key decisions captured: the **align-then-advance phased law**, the residual `Δ_full = pose_des ⊖ c_t` formulation, and the **smoothstep distance gate `g(d)` guaranteeing far-field zero by construction**, plus the four-stage validation. Project-internal rationale; geometric-expert-for-peg-in-hole is standard in the BC literature but no specific `raw/` source backs this page.
