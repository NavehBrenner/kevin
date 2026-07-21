"""Load every checkpoint under outputs/policy/runs/ and report what happens.

The A-3 guard (LAB-110): `PolicyConfig.use_tanh_head` was removed, so every
checkpoint trained before that carries a key the dataclass no longer defines. This
script proves `load_checkpoint`'s drop-unknown-keys shim rescues them all instead of
stranding the project's entire trained-model history.

Run from kevin/:  uv run python scripts/dev/verify_checkpoint_compat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.policy import load_checkpoint  # noqa: E402

RUNS = Path("outputs/policy/runs")

failures = 0
for checkpoint in sorted(RUNS.glob("*/checkpoint.pt")):
    try:
        loaded = load_checkpoint(checkpoint)
    except Exception as error:  # noqa: BLE001 — reporting tool: show, don't raise
        failures += 1
        print(f"FAIL  {checkpoint.parent.name:24} {type(error).__name__}: {error}")
        continue
    modality = "vision" if loaded.config.use_vision else "F/T"
    print(f"ok    {checkpoint.parent.name:24} {modality:6} schema={loaded.data_schema_version}")

print(f"\n{failures} failure(s)")
raise SystemExit(1 if failures else 0)
