from __future__ import annotations

import pytest

import litecrawl as lc
from tests.helpers import build_config


@pytest.mark.asyncio()
async def test_frontier_bootstrap_and_claim(tmp_path) -> None:
    config = build_config(tmp_path)
    rules = lc.UrlRules(config.normalize_patterns, config.include_patterns, config.exclude_patterns)
    frontier = lc.SQLiteFrontier(config.sqlite_path)
    try:
        await frontier.initialize()
        await frontier.bootstrap(config.start_urls, rules, config)
        await frontier.cleanup_stale(lc._utcnow(), config)

        claimed = await frontier.claim_ready(1, lc._utcnow())
        assert len(claimed) == 1
        page = claimed[0]
        assert page.norm_url.startswith("https://example.com")

        now = lc._utcnow()
        await frontier.finalize_page(
            page,
            status_code=200,
            content="payload",
            content_type="text/html",
            changed=True,
            now=now,
            config=config,
        )

        dst_id, is_new = await frontier.ensure_page("https://example.com/next", now, config)
        assert is_new is True
        await frontier.record_link(page.page_id, dst_id, lc.LinkKind.REGULAR)
    finally:
        await frontier.close()
