"""Minimal asynchronous cron-friendly Playwright-based targeted web crawler."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import re
import socket
import time
import urllib.robotparser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import aiosqlite
from lxml import html
from playwright.async_api import BrowserContext, Page, Response, Route, async_playwright

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {"User-Agent": "litecrawl/0.2"}
DEFAULT_VIEWPORT = {"width": 2160, "height": 3840}
DEFAULT_BLOCK_RESOURCES = {"image", "font", "media"}
EMPTY_CONTENT_HASH = ""
EMPTY_CONTENT_TYPE = ""

# --- Hooks Definitions ---
PageReadyHook = Callable[[Page, Response | None, str], Awaitable[None]]
LinkExtractHook = Callable[[Page, Response | None, str], Awaitable[list[str]]]
ContentExtractHook = Callable[[Page, Response | None, str], Awaitable[Any]]
DownstreamHook = Callable[[Any, str, str, bool, int], Awaitable[None]]

# Cache for SSRF safety decisions, keyed by hostname
_SSRF_SAFE_CACHE: dict[str, bool] = {}

@dataclass
class PageRecord:
    norm_url: str
    first_seen_time: int
    last_inlink_seen_time: int
    last_crawl_time: int | None
    next_crawl_time: int | None
    error_count: int
    content_hash: str


class RobotsGatekeeper:
    """Simple cache for robots.txt parsing."""
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    async def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain not in self.parsers:
            parser = urllib.robotparser.RobotFileParser()
            robots_url = urlunparse((parsed.scheme, domain, "/robots.txt", "", "", ""))
            
            # Run blocking I/O in executor to avoid freezing the event loop
            try:
                await asyncio.get_running_loop().run_in_executor(None, parser.set_url, robots_url)
                await asyncio.get_running_loop().run_in_executor(None, parser.read)
            except Exception:
                # On error, assume allowed (standard crawler behavior, or strict based on policy)
                # We'll allow strict failure to default to Allow for resilience.
                pass
            self.parsers[domain] = parser
        
        return self.parsers[domain].can_fetch(self.user_agent, url)


def litecrawl(
    sqlite_path: str,
    start_urls: list[str],
    **kwargs
) -> None:
    """Synchronous wrapper for litecrawl_async."""
    asyncio.run(litecrawl_async(sqlite_path, start_urls, **kwargs))


async def litecrawl_async(
    sqlite_path: str,
    start_urls: list[str],
    normalize_patterns: list[dict[str, str]] | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    n_claims: int = 100,
    n_concurrent: int = 10,
    check_robots_txt: bool = True,
    check_ssrf: bool = True,
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
    error_threshold: int = 3,
    processing_timeout_sec: int = 60 * 10,
) -> None:
    """Async entry point for the crawler."""
    if pw_block_resources is None:
        pw_block_resources = DEFAULT_BLOCK_RESOURCES
    
    pw_headers = pw_headers or DEFAULT_HEADERS
    pw_viewport = pw_viewport or DEFAULT_VIEWPORT

    _validate_intervals(
        min_interval_sec=min_interval_sec,
        max_interval_sec=max_interval_sec,
        fresh_factor=fresh_factor,
        stale_factor=stale_factor,
    )

    robots = RobotsGatekeeper(pw_headers.get("User-Agent", "litecrawl")) if check_robots_txt else None

    async with aiosqlite.connect(sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        await _create_schema(db)
        
        # 1. Prune expired URLs
        await _cleanup_retention(db, inlink_retention_sec)
        
        # 2. Bootstrap Start URLs
        await _bootstrap_start_urls(
            db=db,
            start_urls=start_urls,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            check_ssrf=check_ssrf,
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
            # Use a persistent context or recreate per batch? 
            # Per batch is safer for memory leaks in long-running processes.
            context = await browser.new_context(
                viewport=pw_viewport,
                user_agent=pw_headers.get("User-Agent"),
                extra_http_headers=pw_headers,
            )
            
            semaphore = asyncio.Semaphore(n_concurrent)
            
            # Use asyncio.gather for the batch
            tasks = [
                asyncio.create_task(
                    _process_page(
                        record=record,
                        db=db,
                        context=context,
                        semaphore=semaphore,
                        robots=robots,
                        normalize_patterns=normalize_patterns,
                        include_patterns=include_patterns,
                        exclude_patterns=exclude_patterns,
                        check_ssrf=check_ssrf,
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
                        error_threshold=error_threshold,
                    )
                )
                for record in claimed_records
            ]
            await asyncio.gather(*tasks)
            
            await context.close()
            await browser.close()


def normalize_and_validate_url(
    url: str,
    base_url: str | None,
    normalize_patterns: list[dict[str, str]] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> str | None:
    """Normalize and validate a URL."""
    try:
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

        # Re-parse to ensure integrity after regex
        parsed_normalized = urlparse(normalized)
        final_scheme = parsed_normalized.scheme.lower()
        
        if final_scheme not in {"http", "https"} or not parsed_normalized.netloc:
            return None

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

        if include_patterns and not any(re.search(pattern, normalized) for pattern in include_patterns):
            return None

        if exclude_patterns and any(re.search(pattern, normalized) for pattern in exclude_patterns):
            return None

        return normalized
    except Exception:
        return None


async def _is_safe_url(url: str) -> bool:
    """Prevents SSRF by checking for private IP addresses (async, with hostname cache)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False

        # Check cache first
        cached = _SSRF_SAFE_CACHE.get(hostname)
        if cached is not None:
            return cached

        # If it looks like an IP literal, check it directly
        try:
            ip = ipaddress.ip_address(hostname)
            safe = not (ip.is_private or ip.is_loopback or ip.is_link_local)
            _SSRF_SAFE_CACHE[hostname] = safe
            return safe
        except ValueError:
            # Domain name: need DNS resolution (do it via the event loop, not blocking)
            loop = asyncio.get_running_loop()
            try:
                # This is already executed in a thread pool under the hood
                addr_info = await loop.getaddrinfo(hostname, None)
            except socket.gaierror:
                _SSRF_SAFE_CACHE[hostname] = False
                return False  # DNS failure -> unsafe/unusable

            safe = True
            for _, _, _, _, sockaddr in addr_info:
                ip_str = sockaddr[0]
                ip = ipaddress.ip_address(ip_str)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    safe = False
                    break

            _SSRF_SAFE_CACHE[hostname] = safe
            return safe

    except Exception:
        # Any parsing/lookup weirdness => treat as unsafe
        return False


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
    robots: RobotsGatekeeper | None,
    normalize_patterns: list[dict[str, str]] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    check_ssrf: bool,
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
    error_threshold: int,
) -> None:
    norm_url = record.norm_url
    page_finalized = False
    
    try:
        # Pre-flight Validation
        normalized = normalize_and_validate_url(
            url=norm_url,
            base_url=None,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        
        # Security: SSRF
        if check_ssrf and normalized and not await _is_safe_url(normalized):
             normalized = None

        if normalized is None:
            await _finalize_page(db, record, norm_url, None, "", EMPTY_CONTENT_HASH, False, None,
                                 new_interval_sec, min_interval_sec, max_interval_sec, 
                                 fresh_factor, stale_factor, record.error_count) # Soft failure logic
            await db.execute("DELETE FROM pages WHERE norm_url = ?", (norm_url,))
            await db.commit()
            page_finalized = True
            return

        # Handle Normalization Change
        if normalized != norm_url:
            # Delete old
            await db.execute("DELETE FROM pages WHERE norm_url = ?", (norm_url,))
            await db.commit()
            
            # Ensure new. NOTE: We do NOT force claim it if it's already busy.
            new_record = await _ensure_page_record(db, normalized, claim_if_free=True)
            if not new_record:
                # The target URL is already processing by another worker. We just stop.
                page_finalized = True
                return
            
            record = new_record
            norm_url = record.norm_url

        # Robots.txt Check
        if robots:
            can_fetch = await robots.can_fetch(norm_url)
            if not can_fetch:
                # Treated as a persistent error to back off
                await _finalize_page(db, record, norm_url, None, "", record.content_hash, False, None,
                                     new_interval_sec, min_interval_sec, max_interval_sec, 
                                     fresh_factor, stale_factor, record.error_count + 1)
                page_finalized = True
                return

        page = await context.new_page()
        page.set_default_timeout(pw_timeout_ms)

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
        except Exception as exc:
            logger.warning("Failed to navigate %s: %s", norm_url, exc)
        
        is_success = response is not None and 200 <= response.status < 300
        
        if not is_success:
            new_error_count = record.error_count + 1
            if new_error_count < error_threshold:
                # Soft error - backoff without finalization (just reschedule)
                await page.close()
                await _finalize_with_error(db, record, norm_url, new_error_count, 
                                           stale_factor, max_interval_sec, new_interval_sec)
                page_finalized = True
                return
            # Hard error - proceed to finalize with None content
        
        final_content = None
        final_content_type = EMPTY_CONTENT_TYPE
        new_links_found = False
        
        if is_success:
            # Reset error count implicitly via finalization later

            # Check Content-Type and Redirects
            final_content_type = response.headers.get("content-type", EMPTY_CONTENT_TYPE).lower()
            final_normalized = normalize_and_validate_url(
                url=page.url,
                base_url=None,
                normalize_patterns=normalize_patterns,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            
            if final_normalized != norm_url:
                # Handle Redirect
                # 1. Finalize source as stub
                await _finalize_page(
                    db=db, record=record, norm_url=norm_url, content=None, content_type=EMPTY_CONTENT_TYPE,
                    content_hash=EMPTY_CONTENT_HASH, fresh=False, downstream_hook=downstream_hook,
                    new_interval_sec=new_interval_sec, min_interval_sec=min_interval_sec,
                    max_interval_sec=max_interval_sec, fresh_factor=fresh_factor, 
                    stale_factor=stale_factor, final_error_count=0
                )
                page_finalized = True # Logic for source is done
                
                if final_normalized and (not check_ssrf or await _is_safe_url(final_normalized)):
                    # 2. Switch context to new URL
                    record = await _ensure_page_record(db, final_normalized, claim_if_free=True)
                    if not record:
                         # Target busy, abort
                        await page.close()
                        return
                    norm_url = record.norm_url
                    page_finalized = False # Now working on the new URL
                else:
                    # Invalid redirect target
                    await page.close()
                    return

            if pw_scroll_rounds > 0 and "text/html" in final_content_type:
                await _perform_scrolls(page, pw_scroll_rounds, pw_scroll_wait_ms)
            
            if page_ready_hook:
                try:
                    await page_ready_hook(page, response, norm_url)
                except Exception as exc:
                    logger.warning("page_ready_hook failed for %s: %s", norm_url, exc)

            # 3. DISCOVERY
            cached_html_bytes: bytes | None = None
            found_links: list[str] = []

            if link_extract_hook:
                try:
                    found_links = await link_extract_hook(page, response, norm_url)
                except Exception as exc:
                    logger.warning("link_extract_hook failed for %s: %s", norm_url, exc)

            elif "text/html" in final_content_type:
                try:
                    content_str = await page.content()
                    cached_html_bytes = content_str.encode("utf-8")
                    found_links = _extract_links_default(cached_html_bytes)
                except Exception:
                    pass

            if found_links:
                new_links_found = await _batch_insert_links(
                    db=db,
                    source_url=norm_url,
                    raw_links=found_links,
                    normalize_patterns=normalize_patterns,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                    check_ssrf=check_ssrf
                )

            # 4. PAYLOAD
            if content_extract_hook:
                try:
                    final_content = await content_extract_hook(page, response, norm_url)
                except Exception as exc:
                    logger.warning("content_extract_hook failed for %s: %s", norm_url, exc)
            else:
                if cached_html_bytes is not None:
                    final_content = cached_html_bytes
                else:
                    try:
                        final_content = await response.body()
                    except Exception:
                        pass
        
        await page.close()

        # 5. HASH & FINALIZE
        # Robust Freshness: Only update hash if content is present.
        # If content is None (error), we keep the OLD hash to prevent flipping.
        if final_content is not None:
            content_hash = _compute_stable_hash(final_content)
        else:
            content_hash = record.content_hash

        content_changed = (final_content is not None) and (record.content_hash != content_hash)
        is_fresh = new_links_found or content_changed
        
        final_error_count = 0 if is_success else (record.error_count + 1)

        await _finalize_page(
            db=db,
            record=record,
            norm_url=norm_url,
            content=final_content,
            content_type=final_content_type,
            content_hash=content_hash,
            fresh=is_fresh,
            downstream_hook=downstream_hook,
            new_interval_sec=new_interval_sec,
            min_interval_sec=min_interval_sec,
            max_interval_sec=max_interval_sec,
            fresh_factor=fresh_factor,
            stale_factor=stale_factor,
            final_error_count=final_error_count,
        )
        page_finalized = True

    finally:
        # Safety Net: Release processing lock if we crashed before finalization
        if not page_finalized:
            await _release_lock(db, norm_url)


def _extract_links_default(content: bytes) -> list[str]:
    """Helper to extract raw links from HTML bytes using lxml with recovery."""
    try:
        # Use HTMLParser with recover=True for malformed HTML
        parser = html.HTMLParser(recover=True)
        document = html.fromstring(content, parser=parser)
    except Exception:
        return []

    hrefs = [e.get("href") for e in document.xpath("//a[@href]") if e.get("href")]
    actions = [e.get("action") for e in document.xpath("//form[@action]") if e.get("action")]
    frames = [e.get("src") for e in document.xpath("//iframe[@src]") if e.get("src")]
    return hrefs + actions + frames


async def _batch_insert_links(
    db: aiosqlite.Connection,
    source_url: str,
    raw_links: list[str],
    normalize_patterns: list[dict[str, str]] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    check_ssrf: bool,
) -> bool:
    """Batch inserts/updates found links to avoid N+1 queries."""
    
    valid_urls = set()
    for raw in raw_links:
        normalized = normalize_and_validate_url(
            url=raw,
            base_url=source_url,
            normalize_patterns=normalize_patterns,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        if normalized and (not check_ssrf or await _is_safe_url(normalized)):
            valid_urls.add(normalized)
            
    if not valid_urls:
        return False

    # 1. Batch Insert New
    # SQLite executemany with IGNORE is efficient
    insert_data = [(u, EMPTY_CONTENT_HASH) for u in valid_urls]
    cursor = await db.executemany(
        """
        INSERT OR IGNORE INTO pages(
            norm_url, first_seen_time, last_inlink_seen_time,
            last_crawl_time, next_crawl_time, processing_time, error_count, content_hash
        )
        VALUES (?, unixepoch(), unixepoch(), NULL, NULL, NULL, 0, ?)
        """,
        insert_data
    )
    
    inserted_count = cursor.rowcount if cursor.rowcount > 0 else 0
    
    # 2. Batch Update Existing (Retention)
    # We update last_inlink_seen_time for ALL found valid URLs
    placeholders = ",".join("?" for _ in valid_urls)
    await db.execute(
        f"UPDATE pages SET last_inlink_seen_time = unixepoch() WHERE norm_url IN ({placeholders})",
        list(valid_urls)
    )
    
    await db.commit()
    return inserted_count > 0


async def _release_lock(db: aiosqlite.Connection, norm_url: str) -> None:
    """Emergency lock release."""
    try:
        await db.execute("UPDATE pages SET processing_time = NULL WHERE norm_url = ?", (norm_url,))
        await db.commit()
    except Exception:
        pass


async def _finalize_with_error(
    db: aiosqlite.Connection,
    record: PageRecord,
    norm_url: str,
    error_count: int,
    stale_factor: float,
    max_interval_sec: int,
    new_interval_sec: int,
) -> None:
    """Handles soft errors (retry logic) without invoking downstream hooks."""
    prev_interval = (
        record.next_crawl_time - record.last_crawl_time
        if record.last_crawl_time is not None and record.next_crawl_time is not None
        else new_interval_sec
    )
    next_interval = int(min(prev_interval * stale_factor, max_interval_sec))
    
    await db.execute(
        """
        UPDATE pages 
        SET processing_time = NULL, error_count = ?, next_crawl_time = unixepoch() + ?
        WHERE norm_url = ?
        """,
        (error_count, next_interval, norm_url)
    )
    await db.commit()


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
    final_error_count: int,
) -> None:
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
        try:
            await downstream_hook(content, content_type, norm_url, fresh, final_error_count)
        except Exception as exc:
            logger.error("downstream_hook error for %s: %s", norm_url, exc)

    await db.execute(
        """
        INSERT INTO pages(
            norm_url, first_seen_time, last_inlink_seen_time, 
            last_crawl_time, next_crawl_time, processing_time, error_count, content_hash
        )
        VALUES (?, unixepoch(), unixepoch(), unixepoch(), unixepoch() + ?, NULL, ?, ?)
        ON CONFLICT(norm_url) DO UPDATE SET
            last_crawl_time = excluded.last_crawl_time,
            next_crawl_time = excluded.next_crawl_time,
            processing_time = NULL,
            error_count = excluded.error_count,
            content_hash = excluded.content_hash;
        """,
        (norm_url, next_interval, final_error_count, content_hash),
    )
    await db.commit()


async def _ensure_page_record(
    db: aiosqlite.Connection, norm_url: str, claim_if_free: bool
) -> PageRecord | None:
    """
    Ensures a record exists. 
    If claim_if_free is True, attempts to lock it (processing_time=now).
    Returns None if the record exists but is ALREADY locked by another worker.
    """
    # 1. Ensure existence
    await db.execute(
        """
        INSERT OR IGNORE INTO pages(
            norm_url, first_seen_time, last_inlink_seen_time,
            last_crawl_time, next_crawl_time, processing_time, error_count, content_hash
        )
        VALUES (?, unixepoch(), unixepoch(), NULL, NULL, NULL, 0, ?)
        """,
        (norm_url, EMPTY_CONTENT_HASH),
    )
    await db.commit()

    if claim_if_free:
        # 2. Attempt to claim ONLY if processing_time is NULL
        cursor = await db.execute(
            """
            UPDATE pages 
            SET processing_time = unixepoch() 
            WHERE norm_url = ? AND processing_time IS NULL
            """,
            (norm_url,)
        )
        if cursor.rowcount == 0:
            await db.commit()
            # It was already locked by someone else, or didn't exist (handled by insert)
            # Check if it's locked
            async with db.execute("SELECT processing_time FROM pages WHERE norm_url = ?", (norm_url,)) as cur:
                row = await cur.fetchone()
                if row and row["processing_time"] is not None:
                    return None # Busy
        await db.commit()

    # 3. Fetch
    query = """
    SELECT norm_url, first_seen_time, last_inlink_seen_time, 
           last_crawl_time, next_crawl_time, error_count, content_hash
    FROM pages WHERE norm_url = ?
    """
    cursor = await db.execute(query, (norm_url,))
    row = await cursor.fetchone()
    
    if not row:
        return None

    return PageRecord(
        norm_url=row["norm_url"],
        first_seen_time=row["first_seen_time"],
        last_inlink_seen_time=row["last_inlink_seen_time"],
        last_crawl_time=row["last_crawl_time"],
        next_crawl_time=row["next_crawl_time"],
        error_count=row["error_count"],
        content_hash=row["content_hash"],
    )


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
    # Enable WAL for better concurrency
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS pages (
          norm_url              TEXT PRIMARY KEY,
          first_seen_time       INTEGER NOT NULL,
          last_inlink_seen_time INTEGER NOT NULL,
          last_crawl_time       INTEGER NULL,
          next_crawl_time       INTEGER NULL,
          processing_time       INTEGER NULL,
          error_count           INTEGER NOT NULL DEFAULT 0,
          content_hash          TEXT NOT NULL
        );
        """
    )
    await db.execute("CREATE INDEX IF NOT EXISTS idx_pages_last_inlink_seen ON pages(last_inlink_seen_time);")
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
    normalize_patterns: list[dict[str, str]] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    check_ssrf: bool,
) -> None:
    valid_starts = []
    for raw in start_urls:
        norm = normalize_and_validate_url(raw, None, normalize_patterns, include_patterns, exclude_patterns)
        if norm and (not check_ssrf or await _is_safe_url(norm)):
            valid_starts.append(norm)

    if not valid_starts:
        return

    insert_data = [(u, EMPTY_CONTENT_HASH) for u in valid_starts]
    await db.executemany(
        """
        INSERT INTO pages(
            norm_url, first_seen_time, last_inlink_seen_time,
            last_crawl_time, next_crawl_time, processing_time, error_count, content_hash
        )
        VALUES (?, unixepoch(), unixepoch(), NULL, NULL, NULL, 0, ?)
        ON CONFLICT(norm_url) DO UPDATE SET
            last_inlink_seen_time = unixepoch()
        """,
        insert_data
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
    # We select timed-out rows
    cursor = await db.execute(
        """
        SELECT norm_url, last_crawl_time, next_crawl_time
        FROM pages
        WHERE processing_time IS NOT NULL AND processing_time < ?
        """,
        (cutoff,),
    )
    rows = await cursor.fetchall()
    
    # We can batch update these too, but logic requires calculating interval per row.
    # Given recovery is rare, loop is acceptable here.
    for row in rows:
        last_crawl = row["last_crawl_time"]
        next_crawl = row["next_crawl_time"]
        
        prev_interval = (next_crawl - last_crawl) if (last_crawl and next_crawl) else new_interval_sec
        next_interval_val = min(int(prev_interval * stale_factor), max_interval_sec)
        
        await db.execute(
            """
            UPDATE pages
            SET processing_time = NULL,
                next_crawl_time = unixepoch() + ?,
                last_crawl_time = last_crawl_time
            WHERE norm_url = ?
            """,
            (next_interval_val, row["norm_url"]),
        )
    await db.commit()


async def _claim_pages(db: aiosqlite.Connection, limit: int) -> list[PageRecord]:
    await db.execute("BEGIN IMMEDIATE;")
    cursor = await db.execute(
        """
        SELECT norm_url, first_seen_time, last_inlink_seen_time, 
               last_crawl_time, next_crawl_time, error_count, content_hash
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
            error_count=row["error_count"],
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


def _compute_stable_hash(payload: Any) -> str:
    if payload is None:
        data = b""
    elif isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        try:
            data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        except Exception:
            data = str(payload).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


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


async def _perform_scrolls(page: Page, rounds: int, wait_ms: int) -> None:
    try:
        previous_height = await page.evaluate("document.body.scrollHeight")
    except Exception:
        return

    for _ in range(max(rounds, 0)):
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(wait_ms)
            
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == previous_height:
                break
            previous_height = new_height
        except Exception:
            break


__all__ = ["litecrawl", "litecrawl_async", "normalize_and_validate_url"]