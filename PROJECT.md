# litecrawl – Minimal asynchronous Playwright-based web crawler

A single-file, cron-friendly crawler with its own SQLite frontier, Playwright rendering (with scrolling), robots.txt compliance, simple adaptive scheduling, and a pluggable content cleaner.

* **Language/OS:** Python ≥ 3.10 (3.11 recommended), Linux.
* **License:** Apache-2.0.
* **Default User-Agent:** `litecrawl/0.1`.

---

## Public API

```python
def litecrawl(

    # Initialization
    sqlite_path: str,
    start_urls: list[str],

    # URL gating
    normalize_patterns: list[dict],        # [ { "pattern": str, "replace": str } ]
                                           # all patterns applied to all URLs using re.sub(), may use regex groups
    include_patterns: list[str],           # [ "pattern": str ]
                                           # minimum one required, URLs MUST match minimum one using re.search()
    exclude_patterns: list[str],           # [ "pattern": str ]
                                           # any number accepted, URLs must NOT match any using re.search()

    # HTTP
    headers: dict[str, str] = {"User-Agent": "litecrawl/0.1"},

    # Work distribution
    n_claims: int = 100,                   # max rows claimed per run
    n_concurrent: int = 10,                # max concurrent claims (pool)

    # Playwright
    pw_scroll_rounds: int = 1,             # scroll-to-bottom for dynamic loading
    pw_scroll_wait_ms: int = 800,
    pw_timeout_ms: int = 15000,            # per-page hard timeout (ms)
    pw_viewport: dict = {"width": 2160, "height": 3840},
    pw_respect_robots: bool = True,
    pw_block_media: bool = True,           # abort "image", "font", "media"

    # Content
    content_cleaner: Callable[[str, str, str], str] =
        lambda content, content_type, url: content,

    # Scheduling
    first_interval_sec: int = 60*60*24,    # default `prev_interval` in seconds for new page, must be a positive integer
    min_interval_sec: int = 60*60,         # hard lower interval boundary in seconds, must be a positive integer
    max_interval_sec: int = 60*60*24*7,    # hard upper interval boundary in seconds, must be a positive integer,
                                           # and >= min_interval_sec
    change_factor: float = 0.2,            # multiplicator when changed == True
    no_change_factor: float = 2.0          # multiplicator when changed == False

    # Safeguard
    stale_timeout_sec: 60*10               # time in seconds before processing goes stale and is cleaned

) -> None:
    ...
```

---

## Behavior

### 1) Bootstrap

* Create the SQLite schema if missing.
* Insert `start_urls` as pages idempotently:

  1. normalize via `normalize_patterns` (regex pipeline; group replacements allowed),
  2. accept only if it matches **≥1 include** and **0 excludes**.

### 2) Claiming (cron-safe)

* Claim rows where `next_fetch_time <= NOW()`and `ORDER BY next_fetch_time ASC` limited by `n_concurrent`.
* Set `processing_time = NOW()` on the claimed rows.
* Add rows to an async Playwright pool for processing.
* Once a row is processed, set `processing_time = NULL, last_fetch_time = NOW(), next_fetch_time = NOW() + interval`.
* Update the pool by claiming new rows until `n_claims` have been reached for the current run.
* At any point, the number of pages with `processing_time NOT NULL` must not exceed `n_concurrent`.

### 3) Fetch (Playwright)

* Apply viewport, timeout, global `headers`.

* If `pw_respect_robots=True`, check robots.txt and skip if disallowed (see status_code semantics).

* If `pw_block_media=True`, block heavy assets:

  ```python
  await page.route('**/*', lambda route:
      route.abort() if route.request.resource_type in ('image','font','media')
      else route.continue_()
  )
  ```

* Perform `pw_scroll_rounds` scrolls with `pw_scroll_wait_ms` between; wait for network idle.

**status_code semantics:**

* `NULL` → never fetched yet
* `-1`   → timeout / network error
* `-2`   → robots disallow (request not made)
* `-3`   → fetch skipped by policy (reserved, optional)
* Otherwise → actual HTTP status code (e.g., 200, 301, 404, …)

### 4) Redirects & canonicals

* If the fetch resulted in an HTTP redirect (30x), normalize the final target URL, and if it validates, record a link
  `links(kind='redirect')` from the source to the normalized final target. **Do not store content or extract further links** for the source.
* If the fetched page (HTTP 200) declares a different canonical URL, treat it exactly like a redirect target: normalize, validate, record a link `links(kind='canonical')`, and **do not store content or extract further links** for the source.

### 5) Link extraction

* If the content is HTML and non-redirect/non-canonical, extract `a[href]`, `form[action]` and `iframe[src]` as links.
* Convert any relative URL to absolute using current URL.
* Keep only http/https, convert protocol to lowercase.
* Remove any port :80 URL part.
* Remove any URL fragments (#…).

### 6) Normalize & validate discovered links

* For every discovered link including redirects and canonical:

  * run **normalize** (regex substitutions applied in given order; may use capture groups),
  * then **validate** (regex searches):

    * valid iff it matches **≥1 include** and **0 excludes**,
    * otherwise **drop**.
* Only **valid** targets are created (idempotently) in `pages` and recorded in `links`.

### 7) Change detection (strictly ordered)

1. Initially assume `changed = False`.
2. If at least one **new valid** normalized link target (not previously in `pages`) was found, then `changed = True`.
3. Or if `new_cleaned_content != old_cleaned_content` (output of `content_cleaner(content, content_type, url) -> str`), then `changed = True`.


### 8) Scheduling

Let:

* `prev_interval = (next_fetch_time - last_fetch_time)` if both existed before this fetch;
  otherwise `prev_interval = first_interval_sec`.

Choose factor:

* if `changed == True`  → `factor = change_factor` (default **0.2**; “reset fast”)
* if `changed == False` → `factor = no_change_factor` (default **2.0**; “back off slowly”)

Compute and clamp:

```
next_interval = prev_interval * factor
if next_interval < min_interval_sec: next_interval = min_interval_sec
if next_interval > max_interval_sec: next_interval = max_interval_sec
```

Finally, when the page has been fetched and processed:

* `processing_time = NULL`
* `last_fetch_time = NOW()`
* `next_fetch_time  = NOW() + next_interval`

### 9) Stale cleanup (at run start)

Rows with `processing_time < NOW() − stale_timeout` are reset (`processing_time = NULL`) and their `next_fetch_time` are calculated with the exact formula used when no change is registered to ensure faulty pages do not continously clog the pipeline.

---

## SQLite Schema

```sql
-- pages: identity + scheduling + cleaned content (no raw bytes)
CREATE TABLE IF NOT EXISTS pages (
  id               INTEGER PRIMARY KEY,
  norm_url         TEXT NOT NULL UNIQUE,
  processing_time  TIMESTAMP NULL,
  last_fetch_time  TIMESTAMP NULL,
  next_fetch_time  TIMESTAMP NULL,
  status_code      INTEGER NULL,              -- see semantics previously defined (-1/-2/-3 or HTTP status)
  content          TEXT NOT NULL DEFAULT '',  -- output of content_cleaner() or "" if redirect/canonical
  content_type     TEXT NULL                  -- e.g., 'text/html', 'application/pdf'
);

-- links: only between normalized, validated source and destination pages
CREATE TABLE IF NOT EXISTS links (
  id           INTEGER PRIMARY KEY,
  src_page_id  INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  dst_page_id  INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  kind         TEXT NOT NULL CHECK (kind IN ('regular','redirect','canonical'))
);

CREATE INDEX IF NOT EXISTS idx_pages_next_fetch ON pages(next_fetch_time);
CREATE INDEX IF NOT EXISTS idx_pages_processing ON pages(processing_time);
CREATE INDEX IF NOT EXISTS idx_links_src ON links(src_page_id);
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst_page_id);
```

---

## Content cleaner contract

```python
def content_cleaner(content: str, content_type: str, url: str) -> str:
    """
    Return ONE deterministic string representing the page content.
    May be plain text or a structured payload (e.g., JSON/YAML dumped as text).
    """
```

* Deterministic (no timestamps/random).
* You may return parsed text, or a compact JSON string (stable key order).
* Action for PDF/other formats is decided in provided cleaner. The default cleaner is **no-op** (returns `content` as given).

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

* `timeout` is the only required safeguard against hangs.
* `n_claims`, `n_concurrent`, and `stale_timeout` parameters prevent overload and overlaps.

---

## Philosophy

* **One file, one public function.**
* **SQLite frontier**, no external queues.
* **Async Playwright rendering** with bandwidth-friendly routing.
* **Normalize → Validate** (includes≥1 and excludes=0).
* **Redirect/canonical** recorded as links (`'redirect'` / `'canonical'`), with **no source content**.
* **Change** = new valid link OR cleaned content string changed.
* **Scheduling** = multiply prev interval (0.2 on change, 2.0 on no change by default), then clamp.
* **Logs** + `status_code` provide diagnostics.