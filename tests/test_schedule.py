import pytest

from litecrawl import PageRecord, _calculate_next_interval, _validate_intervals


def test_next_interval_first_crawl_uses_new_interval():
    record = PageRecord(
        norm_url="https://example.com",
        first_seen_time=0,
        last_inlink_seen_time=0,
        last_crawl_time=None,
        next_crawl_time=None,
        error_count=0,
        content_hash="",
    )
    result = _calculate_next_interval(
        record=record,
        fresh=False,
        new_interval_sec=3600,
        min_interval_sec=300,
        max_interval_sec=7200,
        fresh_factor=0.5,
        stale_factor=2.0,
    )
    assert result == 3600


def test_next_interval_clamps_fresh_to_min_interval():
    record = PageRecord(
        norm_url="https://example.com",
        first_seen_time=0,
        last_inlink_seen_time=0,
        last_crawl_time=1000,
        next_crawl_time=1300,
        error_count=0,
        content_hash="abc",
    )
    result = _calculate_next_interval(
        record=record,
        fresh=True,
        new_interval_sec=3600,
        min_interval_sec=400,
        max_interval_sec=7200,
        fresh_factor=0.5,
        stale_factor=2.0,
    )
    assert result == 400


def test_next_interval_clamps_stale_to_max_interval():
    record = PageRecord(
        norm_url="https://example.com",
        first_seen_time=0,
        last_inlink_seen_time=0,
        last_crawl_time=1000,
        next_crawl_time=1600,
        error_count=0,
        content_hash="abc",
    )
    result = _calculate_next_interval(
        record=record,
        fresh=False,
        new_interval_sec=3600,
        min_interval_sec=300,
        max_interval_sec=1000,
        fresh_factor=0.5,
        stale_factor=2.0,
    )
    assert result == 1000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_interval_sec": 0, "max_interval_sec": 10, "fresh_factor": 0.5, "stale_factor": 1.0},
        {"min_interval_sec": 10, "max_interval_sec": 5, "fresh_factor": 0.5, "stale_factor": 1.0},
        {"min_interval_sec": 10, "max_interval_sec": 20, "fresh_factor": 1.5, "stale_factor": 1.0},
        {"min_interval_sec": 10, "max_interval_sec": 20, "fresh_factor": 0.5, "stale_factor": 0.5},
    ],
)
def test_validate_intervals_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        _validate_intervals(**kwargs)
