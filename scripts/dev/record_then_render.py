"""Two-pass recording: record commands live, then replay offline to render images.

Recording wrist-cam frames live (`--record images`/`all`) puts a MuJoCo render call
in the hot control loop, which is the usual cause of a laggy `--input vision`
session. Since replay reproduces a recorded episode's physics to the step
(`run_episode.py --input <episode>`, documented there), the fix is to split the two
concerns: pass 1 records only the `cmd_*` trajectory (no rendering, so the live loop
stays fast); pass 2 replays those commands headless and renders frames with no
real-time pressure (headless defaults to `--time-factor inf`). Termination
(`episode_terminal_reason`) depends only on physics state, never on rendering, so
both passes end at the identical step and the two outputs line up frame-for-frame.

This script just chains the two `run_episode.py` invocations so you don't have to
compute/pass the shared output dir by hand. Any flag `run_episode.py` accepts is
forwarded to pass 1 (the live/recording pass) — e.g. `--input vision --stereo-calib
... --cameras 0 2`, or `--input scripted --seed 3`. `--record`, `--record-out`, and
`--headless` are controlled by this script, not forwarded.

Run from `kevin/`:

    uv run python scripts/dev/record_then_render.py --input vision --stereo-calib calib.json --cameras 0 2
    uv run python scripts/dev/record_then_render.py --input scripted --seed 3 --wall-seed 7
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_episode import _resolve_record_path  # noqa: E402

_CONTROLLED_FLAGS = {"--record", "--record-out", "--headless"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--record-out",
        default=None,
        metavar="OUT",
        help="Output dir for both passes (episode.npz + imgs/). Auto-numbered under "
        "data/recorded/ if omitted (same scheme as run_episode.py).",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        metavar="N",
        help="Image cadence for the render pass: save a frame every N recorded steps.",
    )
    args, forwarded = parser.parse_known_args()

    for bad in _CONTROLLED_FLAGS:
        if bad in forwarded:
            parser.error(f"{bad} is controlled by this wrapper, not forwarded — drop it.")

    record_dir = _resolve_record_path(args.record_out).parent
    run_episode_py = str(Path(__file__).resolve().parent.parent / "run_episode.py")

    print(f"[1/2] Recording commands (live) -> {record_dir}")
    subprocess.run(
        [
            sys.executable,
            run_episode_py,
            "--record",
            "commands",
            "--record-out",
            str(record_dir),
            *forwarded,
        ],
        check=True,
    )

    print(f"[2/2] Replaying to render images (headless) -> {record_dir / 'imgs'}")
    subprocess.run(
        [
            sys.executable,
            run_episode_py,
            "--headless",
            "--input",
            str(record_dir),
            "--record",
            "images",
            "--record-out",
            str(record_dir),
            "--render-every",
            str(args.render_every),
        ],
        check=True,
    )
    print(f"Done: {record_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
