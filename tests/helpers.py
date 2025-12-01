from __future__ import annotations

from typing import Any

import litecrawl as lc


def build_config(tmp_path: Any, **overrides: Any) -> lc.Config:
    """Return a fully-populated Config instance for tests."""

    cfg = lc.Config(
        sqlite_path=str(tmp_path / "litecrawl.db"),
        start_urls=["https://example.com/"],
        normalize_patterns=[{"pattern": r"/$", "replace": ""}],
        include_patterns=[r"example\.com"],
        exclude_patterns=[],
        headers={"User-Agent": "test-suite"},
        n_claims=10,
        n_concurrent=2,
        pw_scroll_rounds=0,
        pw_scroll_wait_ms=10,
        pw_timeout_ms=1_000,
        pw_viewport={"width": 800, "height": 600},
        pw_respect_robots=False,
        pw_block_media=False,
        content_cleaner=lambda content, _ctype, _url: content,
        first_interval_sec=60,
        min_interval_sec=30,
        max_interval_sec=3600,
        change_factor=0.5,
        no_change_factor=2.0,
        stale_timeout_sec=60,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
