"""Minimal asynchronous Playwright-based web crawler."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlunparse

import aiosqlite
from lxml import html
from playwright.async_api import BrowserContext, Page, Response, Route, async_playwright

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {"User-Agent": "litecrawl/0.1"}
DEFAULT_VIEWPORT = {"width": 2160, "height": 3840}
EMPTY_CONTENT_HASH = ""

PageHook = Callable[[Page], Awaitable[None]]
TransformHook = Callable[[bytes, str, str], str | bytes]
DownstreamHook = Callable[[bytes, str, str, bool], None]


@dataclass
class PageRecord:
    norm_url: str
    last_crawl_time: int | None
    next_crawl_time: int | None
    content_hash: str


def litecrawl(
    sqlite_path: str,
    start_urls: list[str],
    normalize_patterns: list[dict] | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    n_claims: int = 100,
    n_concurrent: int = 10,
    pw_headers: dict[str, str] | None = None,
    pw_scroll_rounds: int = 1,
    pw_scroll_wait_ms: int = 800,
    pw_timeout_ms: int = 15000,
    pw_viewport: dict | None = None,
    pw_block_media: bool = True,
    page_hook: PageHook | None = None,
    transform_hook: TransformHook | None = None,
    downstream_hook: DownstreamHook | None = None,
    new_interval_sec: int = 60 * 60 * 24,
    min_interval_sec: int = 60 * 60,
    max_interval_sec: int = 60 * 60 * 24 * 30,
    fresh_factor: float = 0.2,
    stale_factor: float = 2.0,
    processing_timeout_sec: int = 60 * 10,
) -> None:
    """Run litecrawl once.

    Args:
        sqlite_path: Path to the SQLite database.
        start_urls: Seed URLs inserted idempotently.
        normalize_patterns: Patterns applied via ``re.sub`` for normalization.
        include_patterns: URLs must match at least one pattern if provided.
        exclude_patterns: URLs must not match any pattern if provided.
        n_claims: Maximum pages claimed per run.
        n_concurrent: Concurrent Playwright tasks.
        pw_headers: Headers applied to every request.
        pw_scroll_rounds: Scroll-to-bottom repetitions.
        pw_scroll_wait_ms: Wait between scrolls in milliseconds.
        pw_timeout_ms: Per-page hard timeout in milliseconds.
        pw_viewport: Viewport configuration; defaults to 2160x3840.
        pw_block_media: Abort heavy media requests when True.
        page_hook: Optional async page interaction hook.
        transform_hook: Optional content transformation hook.
        downstream_hook: Optional final hook invoked after processing.
        new_interval_sec: Interval applied after the first crawl.
        min_interval_sec: Minimum interval clamp; must be > 0.
        max_interval_sec: Maximum interval clamp; must be >= min_interval_sec.
        fresh_factor: Multiplier when content is fresh; must be <= 1.0.
        stale_factor: Multiplier when content is stale; must be >= 1.0.
        processing_timeout_sec: Processing safeguard timeout in seconds.

    Raises:
        ValueError: If interval arguments are invalid.
    """
    asyncio.run(
        _litecrawl(
            sqlite_path=sqlite_path,
            start_urls=start_urls,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            n_claims=n_claims,
            n_concurrent=n_concurrent,
            pw_headers=pw_headers or DEFAULT_HEADERS,
            pw_scroll_rounds=pw_scroll_rounds,
            pw_scroll_wait_ms=pw_scroll_wait_ms,
            pw_timeout_ms=pw_timeout_ms,
            pw_viewport=pw_viewport or DEFAULT_VIEWPORT,
            pw_block_media=pw_block_media,
            page_hook=page_hook,
            transform_hook=transform_hook,
            downstream_hook=downstream_hook,
            new_interval_sec=new_interval_sec,
            min_interval_sec=min_interval_sec,
            max_interval_sec=max_interval_sec,
            fresh_factor=fresh_factor,
            stale_factor=stale_factor,
            processing_timeout_sec=processing_timeout_sec,
        )
    )


def normalize_and_validate_url(
    url: str,
    base_url: str | None,
    normalize_patterns: list[dict] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> str | None:
    """Normalize and validate a URL according to crawler settings.

    Args:
        url: The URL to normalize.
        base_url: Base URL used to resolve relative paths.
        normalize_patterns: Patterns applied sequentially via ``re.sub``.
        include_patterns: At least one pattern must match when provided.
        exclude_patterns: No pattern may match when provided.

    Returns:
        A normalized URL or ``None`` when invalid.
    """
    candidate = url.strip()
    base = base_url.strip() if base_url else None

    if base and not urlparse(candidate).scheme:
        candidate = urljoin(base, candidate)

    parsed = urlparse(candidate)
    if not parsed.scheme:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return None

    hostname = parsed.hostname or ""
    port = parsed.port
    if port in (80, 443):
        netloc = hostname
    else:
        netloc = parsed.netloc

    path = parsed.path or "/"
    normalized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))

    if normalize_patterns:
        for pattern in normalize_patterns:
            pattern_value = pattern.get("pattern")
            replace_value = pattern.get("replace", "")
            if pattern_value is None:
                continue
            normalized = re.sub(pattern_value, replace_value, normalized)

    parsed_normalized = urlparse(normalized)
    final_scheme = parsed_normalized.scheme.lower()
    normalized = urlunparse(
        (
            final_scheme,
            parsed_normalized.netloc,
            parsed_normalized.path or "/",
            parsed_normalized.params,
            parsed_normalized.query,
            "",
        )
    )
    if final_scheme not in {"http", "https"} or not parsed_normalized.netloc:
        return None

    if include_patterns and not any(re.search(pattern, normalized) for pattern in include_patterns):
        return None

    if exclude_patterns and any(re.search(pattern, normalized) for pattern in exclude_patterns):
        return None

    return normalized


async def _litecrawl(
    sqlite_path: str,
    start_urls: list[str],
    normalize_patterns: list[dict] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    n_claims: int,
    n_concurrent: int,
    pw_headers: dict[str, str],
    pw_scroll_rounds: int,
    pw_scroll_wait_ms: int,
    pw_timeout_ms: int,
    pw_viewport: dict,
    pw_block_media: bool,
    page_hook: PageHook | None,
    transform_hook: TransformHook | None,
    downstream_hook: DownstreamHook | None,
    new_interval_sec: int,
    min_interval_sec: int,
    max_interval_sec: int,
    fresh_factor: float,
    stale_factor: float,
    processing_timeout_sec: int,
) -> None:
    _validate_intervals(
        min_interval_sec=min_interval_sec,
        max_interval_sec=max_interval_sec,
        fresh_factor=fresh_factor,
        stale_factor=stale_factor,
    )
    async with aiosqlite.connect(sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        await _create_schema(db)
        await _bootstrap_start_urls(
            db=db,
            start_urls=start_urls,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        await _cleanup_processing(
            db=db,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            new_interval_sec=new_interval_sec,
            min_interval_sec=min_interval_sec,
            max_interval_sec=max_interval_sec,
            stale_factor=stale_factor,
            processing_timeout_sec=processing_timeout_sec,
        )
        claimed_records = await _claim_pages(db=db, limit=n_claims)
        if not claimed_records:
            return

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport=pw_viewport,
                user_agent=pw_headers.get("User-Agent"),
                extra_http_headers=pw_headers,
            )
            semaphore = asyncio.Semaphore(n_concurrent)
            tasks = [
                asyncio.create_task(
                    _process_page(
                        record=record,
                        db=db,
                        context=context,
                        semaphore=semaphore,
                        normalize_patterns=normalize_patterns,
                        include_patterns=include_patterns,
                        exclude_patterns=exclude_patterns,
                        pw_scroll_rounds=pw_scroll_rounds,
                        pw_scroll_wait_ms=pw_scroll_wait_ms,
                        pw_timeout_ms=pw_timeout_ms,
                        pw_block_media=pw_block_media,
                        page_hook=page_hook,
                        transform_hook=transform_hook,
                        downstream_hook=downstream_hook,
                        new_interval_sec=new_interval_sec,
                        min_interval_sec=min_interval_sec,
                        max_interval_sec=max_interval_sec,
                        fresh_factor=fresh_factor,
                        stale_factor=stale_factor,
                    )
                )
                for record in claimed_records
            ]
            await asyncio.gather(*tasks)
            await context.close()
            await browser.close()


def _validate_intervals(
    *,
    min_interval_sec: int,
    max_interval_sec: int,
    fresh_factor: float,
    stale_factor: float,
) -> None:
    if min_interval_sec <= 0:
        raise ValueError("min_interval_sec must be > 0")
    if max_interval_sec < min_interval_sec:
        raise ValueError("max_interval_sec must be >= min_interval_sec")
    if fresh_factor > 1.0:
        raise ValueError("fresh_factor must be <= 1.0")
    if stale_factor < 1.0:
        raise ValueError("stale_factor must be >= 1.0")


async def _create_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS pages (
          norm_url         TEXT PRIMARY KEY,
          last_crawl_time  INTEGER NULL,
          next_crawl_time  INTEGER NULL,
          processing_time  INTEGER NULL,
          content_hash     TEXT NOT NULL
        );
        """
    )
    await db.execute("CREATE INDEX IF NOT EXISTS idx_pages_next_crawl ON pages(next_crawl_time);")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_pages_processing ON pages(processing_time);")
    await db.commit()


async def _bootstrap_start_urls(
    db: aiosqlite.Connection,
    start_urls: list[str],
    normalize_patterns: list[dict] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> None:
    for raw_url in start_urls:
        normalized = normalize_and_validate_url(
            url=raw_url,
            base_url=None,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        if normalized is None:
            continue
        await db.execute(
            """
            INSERT OR IGNORE INTO pages(
                norm_url, last_crawl_time, next_crawl_time, processing_time, content_hash
            )
            VALUES (?, NULL, NULL, NULL, ?);
            """,
            (normalized, EMPTY_CONTENT_HASH),
        )
    await db.commit()


async def _cleanup_processing(
    db: aiosqlite.Connection,
    normalize_patterns: list[dict] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    new_interval_sec: int,
    min_interval_sec: int,
    max_interval_sec: int,
    stale_factor: float,
    processing_timeout_sec: int,
) -> None:
    cutoff = int(time.time()) - processing_timeout_sec
    cursor = await db.execute(
        """
        SELECT norm_url, last_crawl_time, next_crawl_time, content_hash
        FROM pages
        WHERE processing_time IS NOT NULL
          AND processing_time < ?
        """,
        (cutoff,),
    )
    rows = await cursor.fetchall()

    for row in rows:
        record = PageRecord(
            norm_url=row["norm_url"],
            last_crawl_time=row["last_crawl_time"],
            next_crawl_time=row["next_crawl_time"],
            content_hash=row["content_hash"],
        )
        normalized = normalize_and_validate_url(
            url=record.norm_url,
            base_url=None,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        if normalized is None:
            await db.execute("DELETE FROM pages WHERE norm_url = ?", (record.norm_url,))
            continue
        if normalized != record.norm_url:
            await db.execute("DELETE FROM pages WHERE norm_url = ?", (record.norm_url,))
            record = await _ensure_page_record(db, normalized, mark_processing=False)

        prev_interval = (
            record.next_crawl_time - record.last_crawl_time
            if record.last_crawl_time is not None and record.next_crawl_time is not None
            else new_interval_sec
        )
        next_interval = min(int(prev_interval * stale_factor), max_interval_sec)
        await db.execute(
            """
            UPDATE pages
            SET processing_time = NULL,
                next_crawl_time = unixepoch() + ?,
                last_crawl_time = last_crawl_time
            WHERE norm_url = ?
            """,
            (next_interval, record.norm_url),
        )
    await db.commit()


async def _claim_pages(db: aiosqlite.Connection, limit: int) -> list[PageRecord]:
    await db.execute("BEGIN IMMEDIATE;")
    cursor = await db.execute(
        """
        SELECT norm_url, last_crawl_time, next_crawl_time, content_hash
        FROM pages
        WHERE (next_crawl_time IS NULL OR next_crawl_time <= unixepoch())
          AND processing_time IS NULL
        ORDER BY next_crawl_time IS NULL DESC, next_crawl_time ASC
        LIMIT ?
        """,
        (limit,),
    )
    rows = await cursor.fetchall()
    records = [
        PageRecord(
            norm_url=row["norm_url"],
            last_crawl_time=row["last_crawl_time"],
            next_crawl_time=row["next_crawl_time"],
            content_hash=row["content_hash"],
        )
        for row in rows
    ]
    if records:
        placeholders = ",".join("?" for _ in records)
        await db.execute(
            f"UPDATE pages SET processing_time = unixepoch() WHERE norm_url IN ({placeholders})",
            [record.norm_url for record in records],
        )
    await db.commit()
    return records


async def _process_page(
    record: PageRecord,
    db: aiosqlite.Connection,
    context: BrowserContext,
    semaphore: asyncio.Semaphore,
    normalize_patterns: list[dict] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    pw_scroll_rounds: int,
    pw_scroll_wait_ms: int,
    pw_timeout_ms: int,
    pw_block_media: bool,
    page_hook: PageHook | None,
    transform_hook: TransformHook | None,
    downstream_hook: DownstreamHook | None,
    new_interval_sec: int,
    min_interval_sec: int,
    max_interval_sec: int,
    fresh_factor: float,
    stale_factor: float,
) -> None:
    async with semaphore:
        await _process_page_inner(
            record=record,
            db=db,
            context=context,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            pw_scroll_rounds=pw_scroll_rounds,
            pw_scroll_wait_ms=pw_scroll_wait_ms,
            pw_timeout_ms=pw_timeout_ms,
            pw_block_media=pw_block_media,
            page_hook=page_hook,
            transform_hook=transform_hook,
            downstream_hook=downstream_hook,
            new_interval_sec=new_interval_sec,
            min_interval_sec=min_interval_sec,
            max_interval_sec=max_interval_sec,
            fresh_factor=fresh_factor,
            stale_factor=stale_factor,
        )


async def _process_page_inner(
    record: PageRecord,
    db: aiosqlite.Connection,
    context: BrowserContext,
    normalize_patterns: list[dict] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    pw_scroll_rounds: int,
    pw_scroll_wait_ms: int,
    pw_timeout_ms: int,
    pw_block_media: bool,
    page_hook: PageHook | None,
    transform_hook: TransformHook | None,
    downstream_hook: DownstreamHook | None,
    new_interval_sec: int,
    min_interval_sec: int,
    max_interval_sec: int,
    fresh_factor: float,
    stale_factor: float,
) -> None:
    norm_url = record.norm_url
    normalized = normalize_and_validate_url(
        url=norm_url,
        base_url=None,
        normalize_patterns=normalize_patterns,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    if normalized is None:
        await db.execute("DELETE FROM pages WHERE norm_url = ?", (norm_url,))
        await db.commit()
        return

    if normalized != norm_url:
        await db.execute("DELETE FROM pages WHERE norm_url = ?", (norm_url,))
        await db.commit()
        record = await _ensure_page_record(db, normalized, mark_processing=True)
        norm_url = record.norm_url

    page = await context.new_page()
    page.set_default_timeout(pw_timeout_ms)
    if pw_block_media:
        await page.route("**/*", _abort_heavy_route)

    response = None
    try:
        response = await page.goto(norm_url)
        await _perform_scrolls(page, pw_scroll_rounds, pw_scroll_wait_ms)
        if page_hook:
            await page_hook(page)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to navigate %s: %s", norm_url, exc)

    final_normalized = normalize_and_validate_url(
        url=page.url if response is not None else norm_url,
        base_url=None,
        normalize_patterns=normalize_patterns,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    if final_normalized is None:
        await db.execute("DELETE FROM pages WHERE norm_url = ?", (norm_url,))
        await db.commit()
        await page.close()
        return

    redirect_inserted = False
    if final_normalized != norm_url:
        redirect_inserted = await _insert_discovered_url(db, final_normalized)
        await _finalize_page(
            db=db,
            record=record,
            norm_url=norm_url,
            content=b"",
            content_type="",
            new_url_found=redirect_inserted,
            transform_hook=transform_hook,
            downstream_hook=downstream_hook,
            new_interval_sec=new_interval_sec,
            min_interval_sec=min_interval_sec,
            max_interval_sec=max_interval_sec,
            fresh_factor=fresh_factor,
            stale_factor=stale_factor,
        )
        record = await _ensure_page_record(db, final_normalized, mark_processing=True)
        norm_url = record.norm_url

    content_bytes = b""
    content_type = ""
    if response is not None:
        content_type = response.headers.get("content-type", "") or ""
        status = response.status
        if 200 <= status < 300:
            content_bytes = await _extract_content(page, response, content_type)
    links_new = False
    if content_type.lower().startswith("text/html") and content_bytes:
        links_new = await _extract_and_store_links(
            db=db,
            source_url=norm_url,
            content=content_bytes,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

    await _finalize_page(
        db=db,
        record=record,
        norm_url=norm_url,
        content=content_bytes,
        content_type=content_type,
        new_url_found=links_new,
        transform_hook=transform_hook,
        downstream_hook=downstream_hook,
        new_interval_sec=new_interval_sec,
        min_interval_sec=min_interval_sec,
        max_interval_sec=max_interval_sec,
        fresh_factor=fresh_factor,
        stale_factor=stale_factor,
    )
    await page.close()


async def _extract_content(page: Page, response: Response, content_type: str) -> bytes:
    if content_type.lower().startswith("text/html"):
        html_content = await page.content()
        return html_content.encode("utf-8")
    return await response.body()


async def _extract_and_store_links(
    db: aiosqlite.Connection,
    source_url: str,
    content: bytes,
    normalize_patterns: list[dict] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> bool:
    try:
        document = html.fromstring(content)
    except Exception:  # noqa: BLE001
        return False

    hrefs = [
        element.get("href")
        for element in document.xpath("//a[@href]")
        if element.get("href") is not None
    ]
    actions = [
        element.get("action")
        for element in document.xpath("//form[@action]")
        if element.get("action") is not None
    ]
    frames = [
        element.get("src")
        for element in document.xpath("//iframe[@src]")
        if element.get("src") is not None
    ]
    urls = hrefs + actions + frames
    new_url_found = False
    for raw_url in urls:
        normalized = normalize_and_validate_url(
            url=raw_url,
            base_url=source_url,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        if normalized is None:
            continue
        inserted = await _insert_discovered_url(db, normalized)
        new_url_found = new_url_found or inserted
    return new_url_found


async def _insert_discovered_url(db: aiosqlite.Connection, norm_url: str) -> bool:
    cursor = await db.execute(
        """
        INSERT OR IGNORE INTO pages(
            norm_url, last_crawl_time, next_crawl_time, processing_time, content_hash
        )
        VALUES (?, NULL, NULL, NULL, ?);
        """,
        (norm_url, EMPTY_CONTENT_HASH),
    )
    await db.commit()
    return cursor.rowcount == 1


async def _finalize_page(
    db: aiosqlite.Connection,
    record: PageRecord,
    norm_url: str,
    content: bytes,
    content_type: str,
    new_url_found: bool,
    transform_hook: TransformHook | None,
    downstream_hook: DownstreamHook | None,
    new_interval_sec: int,
    min_interval_sec: int,
    max_interval_sec: int,
    fresh_factor: float,
    stale_factor: float,
) -> None:
    transformed = await _apply_transform_hook(transform_hook, content, content_type, norm_url)
    content_hash = hashlib.sha256(transformed).hexdigest()

    content_changed = record.content_hash != content_hash
    fresh = new_url_found or content_changed

    next_interval = _calculate_next_interval(
        record=record,
        fresh=fresh,
        new_interval_sec=new_interval_sec,
        min_interval_sec=min_interval_sec,
        max_interval_sec=max_interval_sec,
        fresh_factor=fresh_factor,
        stale_factor=stale_factor,
    )
    if downstream_hook:
        downstream_hook(transformed, content_type, norm_url, fresh)
    await db.execute(
        """
        INSERT INTO pages(norm_url, last_crawl_time, next_crawl_time, processing_time, content_hash)
        VALUES (?, unixepoch(), unixepoch() + ?, NULL, ?)
        ON CONFLICT(norm_url) DO UPDATE SET
            last_crawl_time = excluded.last_crawl_time,
            next_crawl_time = excluded.next_crawl_time,
            processing_time = NULL,
            content_hash = excluded.content_hash;
        """,
        (norm_url, next_interval, content_hash),
    )
    await db.commit()


async def _apply_transform_hook(
    transform_hook: TransformHook | None, content: bytes, content_type: str, norm_url: str
) -> bytes:
    if not transform_hook:
        return content
    try:
        transformed = transform_hook(content, content_type, norm_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("transform_hook failed for %s: %s", norm_url, exc)
        return content
    if isinstance(transformed, bytes):
        return transformed
    if isinstance(transformed, str):
        return transformed.encode("utf-8")
    return str(transformed).encode("utf-8")


def _calculate_next_interval(
    *,
    record: PageRecord,
    fresh: bool,
    new_interval_sec: int,
    min_interval_sec: int,
    max_interval_sec: int,
    fresh_factor: float,
    stale_factor: float,
) -> int:
    if record.last_crawl_time is None:
        return new_interval_sec

    prev_interval = (
        record.next_crawl_time - record.last_crawl_time
        if record.next_crawl_time is not None
        else new_interval_sec
    )
    if fresh:
        return int(max(prev_interval * fresh_factor, min_interval_sec))
    return int(min(prev_interval * stale_factor, max_interval_sec))


async def _ensure_page_record(
    db: aiosqlite.Connection, norm_url: str, mark_processing: bool
) -> PageRecord:
    cursor = await db.execute(
        """
        SELECT norm_url, last_crawl_time, next_crawl_time, content_hash
        FROM pages
        WHERE norm_url = ?
        """,
        (norm_url,),
    )
    row = await cursor.fetchone()
    if row is None:
        await db.execute(
            """
            INSERT INTO pages(
                norm_url, last_crawl_time, next_crawl_time, processing_time, content_hash
            )
            VALUES (?, NULL, NULL, NULL, ?)
            """,
            (norm_url, EMPTY_CONTENT_HASH),
        )
        await db.commit()
        cursor = await db.execute(
            """
            SELECT norm_url, last_crawl_time, next_crawl_time, content_hash
            FROM pages
            WHERE norm_url = ?
            """,
            (norm_url,),
        )
        row = await cursor.fetchone()
    if mark_processing:
        await db.execute(
            "UPDATE pages SET processing_time = unixepoch() WHERE norm_url = ?", (norm_url,)
        )
        await db.commit()
    return PageRecord(
        norm_url=row["norm_url"],
        last_crawl_time=row["last_crawl_time"],
        next_crawl_time=row["next_crawl_time"],
        content_hash=row["content_hash"],
    )


async def _perform_scrolls(page: Page, rounds: int, wait_ms: int) -> None:
    for _ in range(max(rounds, 0)):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(wait_ms)
        await page.wait_for_load_state("networkidle")


async def _abort_heavy_route(route: Route) -> None:
    resource_type = route.request.resource_type
    if resource_type in {"image", "font", "media"}:
        await route.abort()
    else:
        await route.continue_()


__all__ = ["litecrawl", "normalize_and_validate_url"]
