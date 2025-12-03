"""Minimal asynchronous Playwright-based web crawler."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import aiosqlite
from lxml import html
from playwright.async_api import BrowserContext, Page, Response, Route, async_playwright

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {"User-Agent": "litecrawl/0.4"}
DEFAULT_VIEWPORT = {"width": 2160, "height": 3840}
DEFAULT_BLOCK_RESOURCES = {"image", "font", "media"}
EMPTY_CONTENT_HASH = ""

# --- Hooks Definitions ---

# Invoked after navigation and scrolling, but before extraction.
PageReadyHook = Callable[[Page], Awaitable[None]]

# Returns a list of raw URLs found on the page.
LinkExtractHook = Callable[[Page], Awaitable[list[str]]]

# Returns the object/data to be stored and hashed (bytes, str, dict, tuple, etc.).
ContentExtractHook = Callable[[Page], Awaitable[Any]]

# Receives the final payload and freshness status.
DownstreamHook = Callable[[Any, str, str, bool], None]  # (content, content_type, url, fresh)


@dataclass
class PageRecord:
    norm_url: str
    first_seen_time: int
    last_inlink_seen_time: int
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
    pw_block_resources: set[str] | None = None,
    page_ready_hook: PageReadyHook | None = None,
    link_extract_hook: LinkExtractHook | None = None,
    content_extract_hook: ContentExtractHook | None = None,
    downstream_hook: DownstreamHook | None = None,
    new_interval_sec: int = 60 * 60 * 24,
    min_interval_sec: int = 60 * 60,
    max_interval_sec: int = 60 * 60 * 24 * 30,
    fresh_factor: float = 0.2,
    stale_factor: float = 2.0,
    inlink_retention_sec: int = 60 * 60 * 24 * 30,
    processing_timeout_sec: int = 60 * 10,
) -> None:
    """Run litecrawl once.

    Args:
        sqlite_path: Path to the SQLite database.
        start_urls: Seed URLs inserted/updated to prevent expiration.
        normalize_patterns: Patterns applied via ``re.sub`` for normalization.
        include_patterns: URLs must match at least one pattern if provided.
        exclude_patterns: URLs must not match any pattern if provided.
        n_claims: Maximum pages claimed per run.
        n_concurrent: Concurrent Playwright tasks.
        pw_headers: Headers applied to every request.
        pw_scroll_rounds: Scroll-to-bottom repetitions (runs before page_ready_hook).
        pw_scroll_wait_ms: Wait between scrolls in milliseconds.
        pw_timeout_ms: Per-page hard timeout in milliseconds.
        pw_viewport: Viewport configuration.
        pw_block_resources: Resource types to abort. Defaults to images/fonts/media.
        page_ready_hook: Async hook for page interaction (runs after scrolling).
        link_extract_hook: Async hook to extract raw links from the page.
        content_extract_hook: Async hook to extract the payload (HTML, PDF, JSON...).
        downstream_hook: Final sync hook invoked after processing with results.
        new_interval_sec: Interval applied after the first crawl.
        min_interval_sec: Minimum interval clamp.
        max_interval_sec: Maximum interval clamp.
        fresh_factor: Multiplier when content is fresh.
        stale_factor: Multiplier when content is stale.
        inlink_retention_sec: Retention period for URLs not seen as links.
        processing_timeout_sec: Processing safeguard timeout.
    """
    if pw_block_resources is None:
        pw_block_resources = DEFAULT_BLOCK_RESOURCES

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
            pw_block_resources=pw_block_resources,
            page_ready_hook=page_ready_hook,
            link_extract_hook=link_extract_hook,
            content_extract_hook=content_extract_hook,
            downstream_hook=downstream_hook,
            new_interval_sec=new_interval_sec,
            min_interval_sec=min_interval_sec,
            max_interval_sec=max_interval_sec,
            fresh_factor=fresh_factor,
            stale_factor=stale_factor,
            inlink_retention_sec=inlink_retention_sec,
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
    """Normalize and validate a URL according to crawler settings."""
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
    pw_block_resources: set[str],
    page_ready_hook: PageReadyHook | None,
    link_extract_hook: LinkExtractHook | None,
    content_extract_hook: ContentExtractHook | None,
    downstream_hook: DownstreamHook | None,
    new_interval_sec: int,
    min_interval_sec: int,
    max_interval_sec: int,
    fresh_factor: float,
    stale_factor: float,
    inlink_retention_sec: int,
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
        
        # 1. Prune expired URLs (Retention Policy)
        await _cleanup_retention(db, inlink_retention_sec)
        
        # 2. Bootstrap Start URLs
        await _bootstrap_start_urls(
            db=db,
            start_urls=start_urls,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

        # 3. Cleanup stalled processing
        await _cleanup_processing(
            db=db,
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
                        pw_block_resources=pw_block_resources,
                        page_ready_hook=page_ready_hook,
                        link_extract_hook=link_extract_hook,
                        content_extract_hook=content_extract_hook,
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


def _validate_intervals(**kwargs) -> None:
    if kwargs["min_interval_sec"] <= 0:
        raise ValueError("min_interval_sec must be > 0")
    if kwargs["max_interval_sec"] < kwargs["min_interval_sec"]:
        raise ValueError("max_interval_sec must be >= min_interval_sec")
    if kwargs["fresh_factor"] > 1.0:
        raise ValueError("fresh_factor must be <= 1.0")
    if kwargs["stale_factor"] < 1.0:
        raise ValueError("stale_factor must be >= 1.0")


async def _create_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS pages (
          norm_url              TEXT PRIMARY KEY,
          first_seen_time       INTEGER NOT NULL,
          last_inlink_seen_time INTEGER NOT NULL,
          last_crawl_time       INTEGER NULL,
          next_crawl_time       INTEGER NULL,
          processing_time       INTEGER NULL,
          content_hash          TEXT NOT NULL
        );
        """
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_pages_last_inlink_seen ON pages(last_inlink_seen_time);"
    )
    await db.execute("CREATE INDEX IF NOT EXISTS idx_pages_next_crawl ON pages(next_crawl_time);")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_pages_processing ON pages(processing_time);")
    await db.commit()


async def _cleanup_retention(db: aiosqlite.Connection, retention_sec: int) -> None:
    cutoff = int(time.time()) - retention_sec
    await db.execute("DELETE FROM pages WHERE last_inlink_seen_time < ?", (cutoff,))
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
        # Upsert: if it exists, update last_inlink_seen_time so it doesn't expire
        await db.execute(
            """
            INSERT INTO pages(
                norm_url, first_seen_time, last_inlink_seen_time,
                last_crawl_time, next_crawl_time, processing_time, content_hash
            )
            VALUES (?, unixepoch(), unixepoch(), NULL, NULL, NULL, ?)
            ON CONFLICT(norm_url) DO UPDATE SET
                last_inlink_seen_time = unixepoch()
            """,
            (normalized, EMPTY_CONTENT_HASH),
        )
    await db.commit()


async def _cleanup_processing(
    db: aiosqlite.Connection,
    new_interval_sec: int,
    min_interval_sec: int,
    max_interval_sec: int,
    stale_factor: float,
    processing_timeout_sec: int,
) -> None:
    cutoff = int(time.time()) - processing_timeout_sec
    cursor = await db.execute(
        """
        SELECT norm_url, first_seen_time, last_inlink_seen_time, 
               last_crawl_time, next_crawl_time, content_hash
        FROM pages
        WHERE processing_time IS NOT NULL AND processing_time < ?
        """,
        (cutoff,),
    )
    rows = await cursor.fetchall()
    for row in rows:
        record = PageRecord(
            norm_url=row["norm_url"],
            first_seen_time=row["first_seen_time"],
            last_inlink_seen_time=row["last_inlink_seen_time"],
            last_crawl_time=row["last_crawl_time"],
            next_crawl_time=row["next_crawl_time"],
            content_hash=row["content_hash"],
        )
        # Reschedule "stranded" jobs with a stale penalty
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
        SELECT norm_url, first_seen_time, last_inlink_seen_time, 
               last_crawl_time, next_crawl_time, content_hash
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
            first_seen_time=row["first_seen_time"],
            last_inlink_seen_time=row["last_inlink_seen_time"],
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
    **kwargs,
) -> None:
    async with semaphore:
        await _process_page_inner(record=record, db=db, context=context, **kwargs)


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
    pw_block_resources: set[str],
    page_ready_hook: PageReadyHook | None,
    link_extract_hook: LinkExtractHook | None,
    content_extract_hook: ContentExtractHook | None,
    downstream_hook: DownstreamHook | None,
    new_interval_sec: int,
    min_interval_sec: int,
    max_interval_sec: int,
    fresh_factor: float,
    stale_factor: float,
) -> None:
    norm_url = record.norm_url

    # Validate URL before navigation
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

    # Handle if normalization changed the URL
    if normalized != norm_url:
        await db.execute("DELETE FROM pages WHERE norm_url = ?", (norm_url,))
        await db.commit()
        record = await _ensure_page_record(db, normalized, mark_processing=True)
        norm_url = record.norm_url

    page = await context.new_page()
    page.set_default_timeout(pw_timeout_ms)

    # Configure resource blocking
    if pw_block_resources:

        async def _route_handler(route: Route) -> None:
            if route.request.resource_type in pw_block_resources:
                await route.abort()
            else:
                await route.continue_()

        await page.route("**/*", _route_handler)

    # 1. NAVIGATION
    response = None
    try:
        response = await page.goto(norm_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to navigate %s: %s", norm_url, exc)
    
    # Early exit if navigation failed completely
    if response is None:
        await db.execute("DELETE FROM pages WHERE norm_url = ?", (norm_url,))
        await db.commit()
        await page.close()
        return

    # 2. SCROLL & HOOK (Shared Setup)
    # Always scroll first if rounds > 0
    if pw_scroll_rounds > 0:
        await _perform_scrolls(page, pw_scroll_rounds, pw_scroll_wait_ms)
    
    # Then run custom interaction logic
    if page_ready_hook:
        try:
            await page_ready_hook(page)
        except Exception as exc:  # noqa: BLE001
            logger.warning("page_ready_hook failed for %s: %s", norm_url, exc)

    # Check if URL changed (redirects)
    final_normalized = normalize_and_validate_url(
        url=page.url,
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
        # Mark destination as seen/linked just now
        redirect_inserted = await _insert_discovered_url(db, final_normalized)
        # Finalize the original URL record as a redirect stub
        await _finalize_page(
            db=db,
            record=record,
            norm_url=norm_url,
            content=b"",
            content_type="",
            content_hash=EMPTY_CONTENT_HASH,
            fresh=False,
            downstream_hook=None,
            new_interval_sec=new_interval_sec,
            min_interval_sec=min_interval_sec,
            max_interval_sec=max_interval_sec,
            fresh_factor=fresh_factor,
            stale_factor=stale_factor,
        )
        record = await _ensure_page_record(db, final_normalized, mark_processing=True)
        norm_url = record.norm_url

    # 3. DISCOVERY (Link Extraction)
    content_type = response.headers.get("content-type", "").lower()
    cached_html_bytes: bytes | None = None
    found_links: list[str] = []

    # Only extract links if content is HTML
    if "text/html" in content_type:
        if link_extract_hook:
            try:
                found_links = await link_extract_hook(page)
            except Exception as exc:  # noqa: BLE001
                logger.warning("link_extract_hook failed for %s: %s", norm_url, exc)
        else:
            # Default strategy: Get HTML (cache it) and use lxml
            try:
                content_str = await page.content()
                cached_html_bytes = content_str.encode("utf-8")
                found_links = _extract_links_default(cached_html_bytes)
            except Exception:  # noqa: BLE001
                pass

    # Process found links (Normalize and Upsert to DB)
    new_links_found = False
    if found_links:
        new_links_found = await _process_found_links(
            db=db,
            source_url=norm_url,
            raw_links=found_links,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
    if redirect_inserted:
        new_links_found = True

    # 4. PAYLOAD (Content Extraction)
    payload: Any = b""
    
    if content_extract_hook:
        try:
            payload = await content_extract_hook(page)
        except Exception as exc:  # noqa: BLE001
            logger.warning("content_extract_hook failed for %s: %s", norm_url, exc)
    else:
        # Default strategy: Use cached HTML if available, otherwise fetch
        if cached_html_bytes is not None:
            payload = cached_html_bytes
        else:
            try:
                content_str = await page.content()
                payload = content_str.encode("utf-8")
            except Exception:  # noqa: BLE001
                pass

    await page.close()

    # 5. HASH & FINALIZE
    content_hash = _compute_stable_hash(payload)
    content_changed = record.content_hash != content_hash
    is_fresh = new_links_found or content_changed

    await _finalize_page(
        db=db,
        record=record,
        norm_url=norm_url,
        content=payload,
        content_type=content_type,
        content_hash=content_hash,
        fresh=is_fresh,
        downstream_hook=downstream_hook,
        new_interval_sec=new_interval_sec,
        min_interval_sec=min_interval_sec,
        max_interval_sec=max_interval_sec,
        fresh_factor=fresh_factor,
        stale_factor=stale_factor,
    )


def _extract_links_default(content: bytes) -> list[str]:
    """Helper to extract raw links from HTML bytes using lxml."""
    try:
        document = html.fromstring(content)
    except Exception:
        return []

    hrefs = [e.get("href") for e in document.xpath("//a[@href]") if e.get("href")]
    actions = [e.get("action") for e in document.xpath("//form[@action]") if e.get("action")]
    frames = [e.get("src") for e in document.xpath("//iframe[@src]") if e.get("src")]
    return hrefs + actions + frames


async def _process_found_links(
    db: aiosqlite.Connection,
    source_url: str,
    raw_links: list[str],
    normalize_patterns: list[dict] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> bool:
    new_url_found = False
    for raw in raw_links:
        normalized = normalize_and_validate_url(
            url=raw,
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
    # 1. Try to insert new record
    cursor = await db.execute(
        """
        INSERT OR IGNORE INTO pages(
            norm_url, first_seen_time, last_inlink_seen_time,
            last_crawl_time, next_crawl_time, processing_time, content_hash
        )
        VALUES (?, unixepoch(), unixepoch(), NULL, NULL, NULL, ?)
        """,
        (norm_url, EMPTY_CONTENT_HASH),
    )
    if cursor.rowcount > 0:
        await db.commit()
        return True

    # 2. If it existed, update retention timestamp
    await db.execute(
        "UPDATE pages SET last_inlink_seen_time = unixepoch() WHERE norm_url = ?",
        (norm_url,),
    )
    await db.commit()
    return False


def _compute_stable_hash(payload: Any) -> str:
    """Computes a SHA256 hash of the payload regardless of type."""
    if isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        # Fallback for complex objects (dicts, lists).
        # We use json.dumps with sort_keys to ensure deterministic hashing for dicts.
        try:
            data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        except Exception:
            # Absolute fallback if object is not JSON serializable
            data = str(payload).encode("utf-8")
            
    return hashlib.sha256(data).hexdigest()


async def _finalize_page(
    db: aiosqlite.Connection,
    record: PageRecord,
    norm_url: str,
    content: Any,
    content_type: str,
    content_hash: str,
    fresh: bool,
    downstream_hook: DownstreamHook | None,
    new_interval_sec: int,
    min_interval_sec: int,
    max_interval_sec: int,
    fresh_factor: float,
    stale_factor: float,
) -> None:
    # Calculate next crawl interval
    next_interval = _calculate_next_interval(
        record=record,
        fresh=fresh,
        new_interval_sec=new_interval_sec,
        min_interval_sec=min_interval_sec,
        max_interval_sec=max_interval_sec,
        fresh_factor=fresh_factor,
        stale_factor=stale_factor,
    )

    # Invoke downstream
    if downstream_hook:
        try:
            downstream_hook(content, content_type, norm_url, fresh)
        except Exception as exc:  # noqa: BLE001
            logger.error("downstream_hook error for %s: %s", norm_url, exc)

    # Update DB
    await db.execute(
        """
        INSERT INTO pages(
            norm_url, first_seen_time, last_inlink_seen_time, 
            last_crawl_time, next_crawl_time, processing_time, content_hash
        )
        VALUES (?, unixepoch(), unixepoch(), unixepoch(), unixepoch() + ?, NULL, ?)
        ON CONFLICT(norm_url) DO UPDATE SET
            last_crawl_time = excluded.last_crawl_time,
            next_crawl_time = excluded.next_crawl_time,
            processing_time = NULL,
            content_hash = excluded.content_hash;
        """,
        (norm_url, next_interval, content_hash),
    )
    await db.commit()


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
    query = """
    SELECT norm_url, first_seen_time, last_inlink_seen_time, 
           last_crawl_time, next_crawl_time, content_hash
    FROM pages WHERE norm_url = ?
    """
    cursor = await db.execute(query, (norm_url,))
    row = await cursor.fetchone()
    
    if row is None:
        # Edge case: processing a URL that wasn't in DB (e.g., redirect target)
        await db.execute(
            """
            INSERT INTO pages(
                norm_url, first_seen_time, last_inlink_seen_time,
                last_crawl_time, next_crawl_time, processing_time, content_hash
            )
            VALUES (?, unixepoch(), unixepoch(), NULL, NULL, NULL, ?)
            """,
            (norm_url, EMPTY_CONTENT_HASH),
        )
        await db.commit()
        cursor = await db.execute(query, (norm_url,))
        row = await cursor.fetchone()

    if mark_processing:
        await db.execute(
            "UPDATE pages SET processing_time = unixepoch() WHERE norm_url = ?", (norm_url,)
        )
        await db.commit()

    return PageRecord(
        norm_url=row["norm_url"],
        first_seen_time=row["first_seen_time"],
        last_inlink_seen_time=row["last_inlink_seen_time"],
        last_crawl_time=row["last_crawl_time"],
        next_crawl_time=row["next_crawl_time"],
        content_hash=row["content_hash"],
    )


async def _perform_scrolls(page: Page, rounds: int, wait_ms: int) -> None:
    try:
        previous_height = await page.evaluate("document.body.scrollHeight")
    except Exception:  # noqa: BLE001
        return

    for _ in range(max(rounds, 0)):
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(wait_ms)
            
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == previous_height:
                break
            previous_height = new_height
        except Exception:  # noqa: BLE001
            break


__all__ = ["litecrawl", "normalize_and_validate_url"]