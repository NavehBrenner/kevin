# Archived checkpoints — the retention policy, decided once (LAB-114)

**A checkpoint is committed only if it backs a published number and cannot be regenerated.**

Since LAB-114 seeded training (`torch.manual_seed(seed)` in `policy/train.py`), a run is
reproducible from *corpus + `--seed` + git commit* — all three recorded in the run's
`metadata.json`, which is committed. Those checkpoints stay gitignored: re-train to get
them back, and check the `checkpoint_sha256` in `metadata.json` to confirm you did.

The two runs here predate that fix. Their weight init and batch shuffling came from OS
entropy, so **no command reproduces them**, and their results are quoted in
`docs/phase-1-results.md`. Losing them would repeat audit finding H-8 — the original
2026-07-07 headline checkpoint is already gone, which is why its 70.0% can no longer be
arbitrated. 716 KB each; git is the cheapest place they can't disappear from.

| Run | What it published |
|---|---|
| `lab101_ft_ar0_ds10` | 2026-07-22 reproduction attempt, `abort_ratio 0` — 46.7% (14/30) |
| `lab101_ft_ar100_ds10` | 2026-07-22 reproduction attempt, `abort_ratio 100` — 46.7% (14/30) |

Each folder is a verbatim copy of `outputs/policy/runs/<name>/` minus `history.png`.
