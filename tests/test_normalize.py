from litecrawl import normalize_and_validate_url


def test_normalize_resolves_relative_and_strips_fragment():
    result = normalize_and_validate_url(
        url="/foo#bar",
        base_url="https://example.com/base",
        normalize_patterns=None,
        include_patterns=None,
        exclude_patterns=None,
    )
    assert result == "https://example.com/foo"


def test_normalize_drops_default_ports_and_lowercases_scheme():
    result = normalize_and_validate_url(
        url="HTTP://example.com:80/path",
        base_url=None,
        normalize_patterns=None,
        include_patterns=None,
        exclude_patterns=None,
    )
    assert result == "http://example.com/path"


def test_normalize_applies_patterns_and_validates():
    result = normalize_and_validate_url(
        url="https://example.com/path",
        base_url=None,
        normalize_patterns=[{"pattern": "example\\.com", "replace": "example.org"}],
        include_patterns=["example\\.org"],
        exclude_patterns=["forbidden"],
    )
    assert result == "https://example.org/path"


def test_normalize_respects_include_and_exclude():
    allowed = normalize_and_validate_url(
        url="https://example.com/allowed",
        base_url=None,
        normalize_patterns=None,
        include_patterns=["allowed"],
        exclude_patterns=["blocked"],
    )
    blocked = normalize_and_validate_url(
        url="https://example.com/blocked",
        base_url=None,
        normalize_patterns=None,
        include_patterns=["blocked"],
        exclude_patterns=["blocked"],
    )
    assert allowed is not None
    assert blocked is None
