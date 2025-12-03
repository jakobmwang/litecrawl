# litecrawl

Minimal asynchronous cron-friendly Playwright-based targeted web crawler.

`litecrawl` is a single-file library built around `litecrawl()` / `litecrawl_async()` with all state in SQLite. It plays nicely with cron, job runners, or multiple workers sharing the same database.

## What it does
- Crawls with headless Chromium using sensible defaults (viewport, headers, timeouts, resource blocking).
- Keeps the frontier in SQLite with atomic claiming so multiple processes can cooperate safely.
- Targets URLs via include/exclude/normalize patterns plus SSRF and robots.txt guard rails.
- Adapts crawl frequency based on freshness (new links or content hash changes) vs. staleness.
- Exposes hooks for page prep, link extraction, content extraction, and downstream handling.

## Installation

```bash
pip install litecrawl
# or
uv add litecrawl

# Install the Chromium browser bundle for Playwright
python -m playwright install chromium
```

## Quick start

```python
from litecrawl import litecrawl

litecrawl(
    sqlite_path="crawl.db",
    start_urls=["https://news.ycombinator.com/"],
    include_patterns=[r"ycombinator\.com"],
    n_claims=50,        # pages to process per run
    n_concurrent=5,     # browser tabs at a time
)
```

Run it on a schedule (example cron entry, once a minute with a 10m safety timeout):

```bash
* * * * * /usr/bin/timeout 10m /usr/bin/python /path/to/crawler.py >> /var/log/crawl.log 2>&1
```

## Hooks
`litecrawl` handles navigation, scheduling, and link discovery. You own the business logic via hooks:

```python
from pathlib import Path
from litecrawl import litecrawl_async


async def page_ready_hook(page, response, url):
    # Click or wait for client-side content before parsing
    try:
        await page.click("#accept-cookies", timeout=1000)
    except Exception:
        pass


async def downstream_hook(content, content_type, url, fresh, error_count):
    if fresh and isinstance(content, bytes) and "text/html" in content_type:
        Path("pages").mkdir(exist_ok=True)
        target = Path("pages") / "latest.html"
        target.write_bytes(content)


async def main():
    await litecrawl_async(
        sqlite_path="crawl.db",
        start_urls=["https://example.com/"],
        include_patterns=[r"example\.com"],
        page_ready_hook=page_ready_hook,
        downstream_hook=downstream_hook,
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Hook signatures:
- `page_ready_hook(page: Page, response: Response | None, url: str) -> Awaitable[None]`
- `link_extract_hook(page, response, url) -> Awaitable[list[str]]` (default: lxml-based extractor)
- `content_extract_hook(page, response, url) -> Awaitable[Any]` (default: response body or cached HTML)
- `downstream_hook(content, content_type, url, fresh, error_count) -> Awaitable[None]`

## Key options (defaults in parentheses)
- `start_urls`: seed URLs inserted if missing.
- `normalize_patterns` (`None`): list of `{pattern, replace}` regex rules applied after normalization.
- `include_patterns` / `exclude_patterns` (`None`): regex allow/deny gates for the frontier.
- `n_claims` (100): rows to claim per run; `n_concurrent` (10): Playwright pages at once.
- `check_robots_txt` (True) and `check_ssrf` (True): guard rails before fetching.
- Playwright: `pw_headers` (User-Agent `litecrawl/0.6`), `pw_viewport` (1920x1080), `pw_timeout_ms` (15000), `pw_scroll_rounds` (1), `pw_scroll_wait_ms` (800), `pw_block_resources` (`{"image","font","media"}`).
- Scheduling: `new_interval_sec` (24h), `min_interval_sec` (1h), `max_interval_sec` (30d), `fresh_factor` (0.2), `stale_factor` (2.0), `inlink_retention_sec` (30d), `error_threshold` (3), `processing_timeout_sec` (600s).

## Scheduling & safety
- SQLite stores the queue plus timing metadata; `BEGIN IMMEDIATE` locks ensure cooperative workers.
- URLs are normalized and filtered before insertion; redirects are normalized and deduped.
- SSRF guard rejects private/loopback/link-local IP targets (with hostname caching).
- Robots.txt is cached per domain and checked before fetching when enabled.
- Fresh pages (new links or new content hash) back off to `fresh_factor * interval` but not below `min_interval_sec`; stale pages back off using `stale_factor` up to `max_interval_sec`.
- Stalled processing locks are cleaned up automatically after `processing_timeout_sec`.

`normalize_and_validate_url` is available if you need to pre-process URLs yourself.

## License

MIT
