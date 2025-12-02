from litecrawl import PageRecord, _calculate_next_interval


def test_next_interval_first_crawl_uses_new_interval():
    record = PageRecord(
        norm_url="https://example.com",
        last_crawl_time=None,
        next_crawl_time=None,
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
        last_crawl_time=1000,
        next_crawl_time=1300,
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
        last_crawl_time=1000,
        next_crawl_time=1600,
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
