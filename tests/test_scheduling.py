from __future__ import annotations

from datetime import timedelta

import litecrawl as lc
from tests.helpers import build_config


def test_compute_next_interval_clamps(tmp_path) -> None:
    config = build_config(
        tmp_path,
        min_interval_sec=60,
        max_interval_sec=180,
        change_factor=0.25,
        no_change_factor=3.0,
    )

    assert lc._compute_next_interval(200, True, config) == 60  # clamps to min
    assert lc._compute_next_interval(70, False, config) == 180  # clamps to max


def test_resolve_prev_interval_defaults(tmp_path) -> None:
    config = build_config(tmp_path)
    now = lc._utcnow()
    later = now + timedelta(seconds=120)
    assert lc._resolve_prev_interval(None, None, config.first_interval_sec) == 60
    assert lc._resolve_prev_interval(now, later, config.first_interval_sec) == 120
