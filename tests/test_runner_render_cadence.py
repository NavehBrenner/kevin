"""`_should_render` — the viewer-sync cadence gate (LAB-88).

The physics loop is deterministic regardless of rendering; this only decides *when* to sync
the viewer. Target rate when there's spare wall-time, a hard floor even when behind. Pure
function, no sim.
"""

from __future__ import annotations

import math

from ai_teleop.sim.runner import _should_render

# 50 fps target / 15 fps floor at 500 Hz sim → every 10th / 33rd physics step.
FRAME, FLOOR = 10, 33


def test_never_faster_than_target():
    # Below the target interval → no render, however much spare time.
    assert not _should_render(FRAME - 1, FRAME, FLOOR, slack=1.0)
    assert not _should_render(FRAME - 1, FRAME, FLOOR, slack=math.inf)


def test_renders_at_target_when_ahead():
    # At the target interval with spare wall-time → render.
    assert _should_render(FRAME, FRAME, FLOOR, slack=0.001)


def test_skips_target_when_behind():
    # Past the target but behind wall-time, and not yet at the floor → skip, give physics time.
    assert not _should_render(FRAME, FRAME, FLOOR, slack=-0.001)


def test_floor_wins_even_when_behind():
    # At the floor interval → render regardless of being behind (the viewer never freezes).
    assert _should_render(FLOOR, FRAME, FLOOR, slack=-100.0)


def test_uncapped_is_always_spare():
    assert _should_render(FRAME, FRAME, FLOOR, slack=math.inf)
