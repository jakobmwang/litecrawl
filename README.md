# litecrawl

`litecrawl` is a minimal, asynchronous web crawler designed for targeted data acquisition. It sits in the operational middle ground: more robust than a simple loop of `requests.get()`, but significantly lighter than distributed systems like Scrapy Cluster or Apache Nutch.

It is designed as a **tool, not a system**. It exposes a single, idempotent function that manages its own state via SQLite. It is intended to be invoked periodically (e.g., via cron) to incrementally build and maintain a dataset.

## Design Philosophy

The architecture follows a "crash-only" software philosophy. It assumes the process will eventually terminate—whether by completion, error, or an external timeout—and ensures that the state remains consistent for the next run.

1.  **State Persistence:** All crawl state (URLs, schedules, content hashes, error counts) is stored in a local SQLite database. This allows the crawler to be stopped and restarted at any time without data loss.
2.  **Adaptive Scheduling:** Unlike simple scrapers that hit every URL on every run, `litecrawl` calculates a `next_crawl_time` for each page. It uses `fresh_factor` and `stale_factor` multipliers to revisit frequently changing pages often and stable pages rarely.
3.  **Process Isolation:** By relying on an external scheduler (cron/systemd) and a strict timeout, memory leaks common in long-running browser processes are mitigated at the OS level.
4.  **Resource Efficiency:** It uses `async_playwright` with shared contexts to execute JavaScript only when necessary, blocking bandwidth-heavy resources (images, fonts) by default.

## Installation

Requires Python 3.12+ and Playwright.

```bash
# Recommended (uv)
uv add litecrawl
uv run playwright install chromium

# Or pip
pip install litecrawl
python -m playwright install chromium
```

## Quick Start

Create a Python script (e.g., `crawler.py`) that calls the entry point. The function will initialize the database, claim a batch of URLs, process them, and exit.

```python
from litecrawl import litecrawl

# Define your configuration
litecrawl(
    sqlite_path="company_news.db",
    start_urls=["https://example.com/news"],
    
    # Only follow links matching these patterns
    include_patterns=[r"https://example\.com/news/.*"],
    
    # Normalize URLs to avoid duplicates (e.g., strip tracking params)
    normalize_patterns=[{"pattern": r"\?utm_source=.*", "replace": ""}],
    
    # Operational settings
    n_concurrent=5,      # Parallel tabs
    n_claims=100,        # Pages to process per run
    fresh_factor=0.5,    # Re-crawl rapidly if content changes
    stale_factor=2.0     # Back off if content is static
)
```

### Deployment via Cron

To run the crawler continuously, set up a cron job. We wrap the execution in `timeout` to handle potential browser hangs or memory leaks gracefully.

```bash
# Run every minute. Kills the process if it exceeds 10 minutes.
* * * * * cd /path/to/project && /usr/bin/timeout 10m uv run python crawler.py >> /var/log/crawl.log 2>&1
```

## Configuration Reference

`litecrawl` is configured entirely through arguments passed to the main function. There are no configuration files to manage.

### Targeting & Discovery
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `sqlite_path` | `str` | Path to the SQLite DB. Created automatically if missing. |
| `start_urls` | `list[str]` | Entry points. Always re-injected on every run to ensure the root is alive. |
| `include_patterns` | `list[str]` | Regex list. Only links matching these are added to the DB. |
| `exclude_patterns` | `list[str]` | Regex list. Links matching these are ignored. |
| `normalize_patterns` | `list[dict]` | Regex replacements (`{"pattern": "...", "replace": "..."}`) applied to URLs before storage. |
| `check_robots_txt` | `bool` | If `True` (default), respects robots.txt rules. |
| `check_ssrf` | `bool` | If `True` (default), prevents connections to private/local IP addresses. |

### Scheduling & Operations
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `n_claims` | `int` | Maximum number of pages to process in one execution. |
| `n_concurrent` | `int` | Number of parallel browser tabs. |
| `fresh_factor` | `float` | Multiplier (< 1.0) to reduce interval if content changes (e.g., 0.5 = crawl twice as often). |
| `stale_factor` | `float` | Multiplier (> 1.0) to increase interval if content is static (e.g., 2.0 = wait twice as long). |
| `new_interval_sec` | `int` | Initial crawl interval for newly discovered pages (default: 24h). |
| `min_interval_sec` | `int` | Minimum time between crawls (floor). |
| `max_interval_sec` | `int` | Maximum time between crawls (ceiling). |

### Browser Control
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `pw_headers` | `dict` | HTTP headers (e.g., User-Agent, Cookies). |
| `pw_viewport` | `dict` | Viewport size (default: 1080p). |
| `pw_block_resources` | `set` | Resource types to abort (default: `{"image", "font", "media"}`). |
| `pw_scroll_rounds` | `int` | How many times to scroll to bottom before extracting (for infinite scroll). |
| `pw_timeout_ms` | `int` | Page load timeout in milliseconds. |

## Advanced Usage

`litecrawl` shines when you need precise control over what constitutes "content" and how it flows downstream.

### 1. Separation of Navigation and Payload

By default, the crawler saves the raw HTML body. However, you often want to traverse one set of pages (e.g., listing pages) but only extract data from another set (e.g., article pages).

You can achieve this by combining `include_patterns` (navigation) with a custom `content_extract_hook` (payload).

```python
async def extract_article(page, response, url):
    # If it's a listing page, return None. 
    # The crawler will still find links, but won't save content/hash for this page.
    if "/category/" in url:
        return None 
    
    # If it's an article, extract the data
    return await page.locator("article.main-content").inner_text()

litecrawl(
    ...,
    include_patterns=[r"/category/", r"/article/"],
    content_extract_hook=extract_article
)
```

### 2. Structured Data & Change Detection

`litecrawl` determines if a page is "fresh" by hashing the return value of your extraction hook. 
*   If you return a **Dictionary**, the crawler automatically sorts the keys and dumps it to JSON before hashing. This ensures deterministic hashes.
*   **Crucial:** Do not include timestamps, random numbers, or ad-rotation IDs in your returned dictionary. These will cause the hash to change on every run, tricking the scheduler into thinking the content is always "fresh," leading to aggressive over-crawling.

```python
async def structured_extract(page, response, url):
    """
    Extracts metadata. Returns a dict.
    """
    return {
        "title": await page.title(),
        "price": await page.locator(".price").inner_text(),
        "stock": await page.locator(".stock-status").inner_text(),
        # "scraped_at": time.time()  <-- WRONG! This breaks change detection.
    }

litecrawl(..., content_extract_hook=structured_extract)
```

### 3. The Downstream Hook

This is where your data leaves the crawler. This hook is called *immediately* after the database is updated. It is the ideal place to push data to Kafka, S3, or an API.

```python
async def push_to_api(content, content_type, url, is_fresh, error_count):
    """
    content: The result from your content_extract_hook (e.g., the dict above)
    is_fresh: True if content hash changed OR new links were found.
    """
    if is_fresh and content:
        payload = {
            "url": url,
            "data": content, # This is the clean dict from step 2
            "crawled_at": time.time() # Add timestamps HERE, not in extraction
        }
        await my_api_client.post("/ingest", json=payload)

litecrawl(..., downstream_hook=push_to_api)
```

### 4. Custom Interaction (Login/Cookies)

Use `page_ready_hook` to interact with the page before links or content are touched.

```python
async def handle_popups(page, response, url):
    # Click generic "Accept Cookies" buttons
    try:
        await page.click("#accept-cookies-btn", timeout=2000)
    except:
        pass
        
    # Wait for a specific element to ensure dynamic content loaded
    await page.wait_for_selector(".dynamic-content")

litecrawl(..., page_ready_hook=handle_popups)
```

## Operational Best Practices

1.  **Concurrency:** Keep `n_concurrent` moderate (5-20). SQLite handles concurrency well, but high write contention from hundreds of workers on a single file can lead to locking issues.
2.  **Database Management:** The `sqlite_path` is the only state. Backup this file to back up your crawl frontier.
3.  **Logs:** The tool logs to standard python logging. Redirect stdout/stderr to a file in your cron definition to monitor progress.
4.  **Partitioning:** If you need to crawl 10 distinct websites, it is often better to set up 10 separate cron entries with 10 separate SQLite files rather than one massive monolithic crawl. This isolates failures and simplifies configuration.