from __future__ import annotations

import litecrawl as lc


def test_url_rules_normalize_and_validate() -> None:
    rules = lc.UrlRules(
        normalizers=[
            {"pattern": r"^https://", "replace": "http://"},
            {"pattern": r"/+$", "replace": ""},
        ],
        include_patterns=[r"example\.com"],
        exclude_patterns=[r"/blocked"],
    )

    assert rules.normalize("https://Example.com/path/") == "http://Example.com/path"
    assert rules.is_valid("http://example.com/path")
    assert rules.normalize_and_validate("https://example.com/blocked") is None


def test_prepare_discovered_url_canonicalizes_and_filters() -> None:
    assert (
        lc._prepare_discovered_url("https://example.com/start", "/foo?q=1#frag")
        == "https://example.com/foo?q=1"
    )
    assert lc._prepare_discovered_url("https://example.com/start", "mailto:test@example.com") is None
    assert (
        lc._prepare_discovered_url("https://example.com/start", "HTTP://Example.com:80/index")
        == "http://example.com/index"
    )
