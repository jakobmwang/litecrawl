# litecrawl – Minimal asynchronous Playwright-based web crawler

A single-interface, concurrency-safe crawler with SQLite frontier, Playwright rendering including scrolling and actions, adaptive scheduling, and pluggable content tranformation.

---

## Public API

```python
def litecrawl(

    # Initialization
    sqlite_path: str,
    start_urls: list[str],

    # URL gating
    normalize_patterns: list[dict] | None = None, # [ { "pattern": str, "replace": str } ]
                                                  # all patterns applied to all URLs using re.sub(), may use regex groups
    include_patterns: list[str] | None = None,    # [ "pattern": str ]
                                                  # if any given, URLs MUST match minimum one using re.search()
    exclude_patterns: list[str] | None = None,    # [ "pattern": str ]
                                                  # if any given, URLs must NOT match ANY using re.search()

    # Work distribution
    n_claims: int = 100,                          # pages claimed for processing per run
    n_concurrent: int = 10,                       # concurrent processings per run (connection pool)

    # Playwright
    pw_headers: dict[str, str] | None = None,     # {"User-Agent": "litecrawl/0.1"} if None
    pw_scroll_rounds: int = 1,                    # scroll-to-bottom for dynamic loading
    pw_scroll_wait_ms: int = 800,                 # wait after scrolling
    pw_timeout_ms: int = 15000,                   # per-page hard timeout (ms)
    pw_viewport: dict | None = None,              # {"width": 2160, "height": 3840} if None
    pw_block_media: bool = True,                  # abort "image", "font", "media"

    # Hooks
    page_hook: Callable[[Page], Awaitable[None]] | None = None,  # perform async actions on page object after scroll
    transform_hook: Callable[[bytes, str, str], str | bytes] | None = None,  # transform retrieved content
    downstream_hook: Callable[[bytes, str, str, bool], None] | None = None,  # act on transformed content

    # Scheduling
    new_interval_sec: int = 60*60*24,     # interval in seconds used after first crawl for new pages
    min_interval_sec: int = 60*60,        # hard lower interval boundary in seconds, must be > 0
    max_interval_sec: int = 60*60*24*30,  # hard upper interval boundary in seconds, must be >= min_interval_sec
    fresh_factor: float = 0.2,            # interval multiplicator when fresh == True, must be <= 1.0
    stale_factor: float = 2.0             # interval multiplicator when fresh == False, must be >= 1.0

    # Safeguard
    processing_timeout_sec: int = 60*10   # time in seconds before processing is cleared

) -> None:
    ...
```

---

## SQLite Schema

```sql
-- pages: identity + scheduling
CREATE TABLE IF NOT EXISTS pages (
  norm_url         TEXT PRIMARY KEY,  -- page identity; normalized url
  last_crawl_time  INTEGER NULL,      -- logged last crawl time in unix epoch seconds
  next_crawl_time  INTEGER NULL,      -- scheduled next crawl time in unix epoch seconds
  processing_time  INTEGER NULL,      -- time of claiming in unix epoch seconds, reset to NULL when processed
  content_hash     TEXT NOT NULL,     -- hash of latest content to measure change in content
);

-- indexes: for crawl maturity and safeguard filtering
CREATE INDEX IF NOT EXISTS idx_pages_next_crawl ON pages(next_crawl_time);
CREATE INDEX IF NOT EXISTS idx_pages_processing ON pages(processing_time);
```

---

## Workflow

### 1) Bootstrap

* Create the SQLite schema if missing.
* Normalize and validate `start_urls` (as described in section 7).
* Any valid, normalized `start_urls` are inserted as pages idempotently.

### 2) Clean up old processing

* Any rows with `processing_time` older than `processing_timeout_sec` are reset (`processing_time = NULL`) and their `next_crawl_time` is calculated with the `stale_factor` (as described in section 9) to ensure faulty pages do not continously clog the pipeline.

### 3) Claim pages ready for processing

* Begin `IMMEDIATE` transaction.
* Claim due rows not already being processed, i.e. `WHERE (next_crawl_time IS NULL OR next_crawl_time <= unixepoch()) AND processing_time IS NULL`.
* Prioritize rows never processed before, i.e. `ORDER BY next_crawl_time IS NULL DESC, next_crawl_time ASC`.
* Limit to maximum claims per run, i.e. `LIMIT n_claims`.
* Immediately mark claimed rows as being processed, i.e. `UPDATE ... SET processing_time = unixepoch()`.
* Commit transaction and release lock.
* Process all claimed rows concurrently using async Playwright (as described in section 4) with semaphore limit of `n_concurrent`.
* Once a row is completely processed, update its status (as described in section 9), including `SET processing_time = NULL`.
* Wait for all tasks to complete before exiting.

### 4) Process pages using Playwright pool

* Initialize Playwright (single shared browser instance).
* Then, for each page, first perform sanity check by normalizing the existing `norm_url` (see section 7 for details).
* If the normalized `norm_url` is different from the original `norm_url`:
  * `DELETE FROM pages` for `norm_url`.
  * Continue processing the page with the normalized `norm_url` as the new `norm_url`.
* If `norm_url` (original or replaced with normalized) then does not validate:
  * `DELETE FROM pages` for `norm_url` and end page processing (remove from processing pool).
* Then, create a new page object and apply `pw_viewport`, `pw_timeout_ms`, and global `pw_headers`.
* If `pw_block_media == True`, block heavy assets:

  ```python
  await page.route('**/*', lambda route:
      route.abort() if route.request.resource_type in ('image','font','media')
      else route.continue_()
  )
  ```
* Navigate to `norm_url` (`await page.goto(norm_url)`).
* Perform `pw_scroll_rounds` scrolls with `pw_scroll_wait_ms` between; wait for network idle.
* If `page_hook` is given, apply it for any custom interaction.

### 5) Handle HTTP status and redirects

* The final `page.url` is normalized, and if it is different from `norm_url`, handle it as a redirect:
  * First, content for `norm_url` is considered empty (i.e. `b""`).
  * Second, the normalized `page.url` is considered a (the only) link found at `norm_url`.
  * Then, immediately finalize processing of the claimed row for `norm_url` (as described in section 9).
  * Finally, continue processing the page with the normalized `page.url` as the new `norm_url`.
* If HTTP status is not OK, content for the normalized `page.url` is considered empty.
* If the normalized `page.url` does not validate:
  * `DELETE FROM pages` (if it exists) and end page processing (remove from processing pool).

### 6) Extract links and parse content

* If the content-type header `.startswith('text/html')`, use lxml to extract URLs from `a[href]`, `form[action]` and `iframe[src]`.
* Normalize extracted URLs (see section 7).
* If URLs validate, insert idempotently into `pages` using default values.
* If `transform_hook` given, apply with page content as `bytes` along with content type and normalized URL.
* After application, normalize page content to `bytes`, if returned as `str`.

### 7) Normalize and validate URLs

* Apply in any case where an URL could potentially be non-normal or invalid, e.g. for claimed pages, redirects, and extracted links:

  * Convert relative URLs to absolute using source normalized URL as base, if necessary.
  * Convert protocol to lowercase, keep only `http`/`https`.
  * Remove any port `:80` or `:443`.
  * Remove any URL fragments (`#...`).
  * Apply `normalize_patterns` using `re.sub()` in given order; may use capture groups.
  * Validate URL, keep only if valid:
    * Must match at least one of `include_patterns`, if any given.
    * Must not match any `exclude_patterns`, if any given.

### 8) Detect fresh content (strictly ordered)

1. Initially assume that the processed page has `fresh = False`.
2. If at least one new valid normalized URL (not already in `pages`) was found (link or redirect), then `fresh = True`.
3. Or if `content_hash` from row in `pages` is different from a sha256 hash of the crawled (and transformed) page content, then `fresh = True`.

### 9) Schedule and finalize processing

* After processing, schedule `next_crawl_time` based on `last_crawl_time` and `fresh`.
* If `last_crawl_time IS NULL`, set `next_interval_sec = new_interval_sec`, else:
  * Calculate `prev_interval_sec = next_crawl_time - last_crawl_time`.
  * If `fresh == True`, then `next_interval_sec = MAX(prev_interval_sec * fresh_factor, min_interval_sec)`.
  * Else if `fresh == False`, then `next_interval_sec = MIN(prev_interval_sec * stale_factor, max_interval_sec)`.
* Then, if `downstream_hook` given, apply.
* Finally, when the page has been processed, `UPSERT`:
  * `norm_url`
  * `last_crawl_time = unixepoch()`
  * `next_crawl_time = unixepoch() + next_interval_sec`
  * `processing_time = NULL`
  * `content_hash = sha256(transformed_content)`

---

## Contracts

```python
def normalize_and_validate_url(
    url: str,
    base_url: str | None,
    normalize_patterns: list[dict] | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> str | None:
    ...

def page_hook(page: Page) -> Awaitable[None]:
    """
    Perform async actions on page after scrolling, before content extraction.
    Example: Click cookie banner, expand collapsed content.
    If triggering network activity, end with appropriate wait.
    """
    ...

def transform_hook(content: bytes, content_type: str, url: str) -> str | bytes:
    """
    Transform content before freshness detection.
    Example: Strip boilerplate from text/html, pass through PDFs, discard other types.
    Must be deterministic (no timestamps or randomness).
    Return "", or b"" to treat as empty (or discarded) content.
    Any returned str is converted to bytes (UTF-8) in postprocessing.
    """
    ...

def downstream_hook(content: bytes, content_type: str, url: str, fresh: bool) -> None:
    """
    Final hook called after processing with transformed content.
    Example: Write to storage, update database, send notifications.
    """
    ...
```

---

## Cron operation (recommended)

Import litecrawl module in example.py and call litecrawl() with desired parameters.

```python
# example.py
from litecrawl import litecrawl

litecrawl(...)
```

Run example.py every minute with a hard wall-clock limit:

```bash
* * * * * /usr/bin/timeout 10m /usr/bin/python3 /opt/example.py >> /var/log/example-litecrawl.log 2>&1
```

Tips:
* Synchronize `/usr/bin/timeout [10m]` and `processing_timeout_sec=[60*10]` for full safeguard against hangs, overload and overlaps.
* Adjust `n_claims` and `n_concurrent` to optimize performance.

---

## Philosophy

* **One file, one public function.**
* **SQLite frontier**, no external queues.
* **Simple batch processing**: Claim n_claims rows atomically, process with n_concurrent semaphore.
* **Async Playwright rendering** with bandwidth-friendly routing.
* **Normalize, then validate** (includes≥1 and excludes=0).
* **Freshness detection** = new valid link OR cleaned content hash changed.
* **Scheduling** = next interval equals previous interval multiplied with factor, then clamp.
* **Logs** provide diagnostics, no built-in status.

---