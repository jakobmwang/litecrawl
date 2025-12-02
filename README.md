# litecrawl

**A minimal, asynchronous, concurrency-safe web crawler.**

`litecrawl` is a single-interface Python library designed to crawl the modern web using Playwright and SQLite. It abandons complex distributed queues and heavy frameworks in favor of a single file, a single function, and a single database connection.

---

## The Philosophy

### 1. A Tool, Not a System
Most crawling frameworks require you to inherit classes, define spiders, configure middleware settings, and manage a long-running process (daemon).

`litecrawl` is a **tool**. It exposes one public function: `litecrawl()`.
*   Need to crawl a news site every 10 minutes? Write a script and put it in `cron`.
*   Need to scrape a competitor once a day? Write a script and put it in `cron`.
*   Need to process 5 different domains with different rules? Run 5 different scripts.

There is no "master process." The state is entirely contained in the SQLite database.

### 2. The Database is the Queue
External message brokers (Redis, RabbitMQ) add unnecessary complexity for datasets under a few million pages. `litecrawl` uses SQLite as both the frontier (URL queue) and the state machine.
*   **Concurrency Safe:** Multiple processes can run `litecrawl()` against the same SQLite file simultaneously. The library uses `IMMEDIATE` transactions to claim rows atomically.
*   **Persistent:** If the process crashes, the state is saved. If the machine reboots, the schedule remains.

### 3. Render First, Optimize Later
The modern web is dynamic. Static HTML parsers often miss content loaded via hydration or AJAX. `litecrawl` assumes **Playwright** is necessary by default.
*   It manages a browser pool automatically.
*   It handles aggressive resource blocking (fonts, images, media) to save bandwidth.
*   It handles scrolling and network idle waits out of the box.

### 4. Adaptive Scheduling
Not all pages update at the same rate. `litecrawl` implements an adaptive scheduling algorithm:
*   **Fresh Content:** If a page changes (hash mismatch) or contains new links, the crawl interval decreases (visits become more frequent).
*   **Stale Content:** If a page is static, the crawl interval increases (visits become less frequent), saving resources.

---

## Installation

```bash
uv add litecrawl
# or
pip install litecrawl
```

You must also install the Playwright browsers:

```bash
uv run playwright install chromium
```

---

## Usage

### Minimal Example

Create a file (e.g., `crawler.py`):

```python
import asyncio
from litecrawl import litecrawl

def main():
    litecrawl(
        sqlite_path="my_crawl.db",
        start_urls=["https://news.ycombinator.com/"],
        include_patterns=[r"ycombinator\.com"],
        n_claims=50,       # Process 50 pages per run
        n_concurrent=5,    # Open 5 browser tabs at a time
    )

if __name__ == "__main__":
    main()
```

### Scheduled Execution (Recommended)

Since `litecrawl` is designed to be a tool, you should schedule it using your system's scheduler (Cron, Systemd timers, Airflow).

**Example Cron Entry (Run every minute):**
```bash
* * * * * /usr/bin/timeout 10m /usr/bin/python3 /path/to/crawler.py >> /var/log/crawl.log 2>&1
```

*Note: The `timeout` command is recommended to prevent process overlaps if a run hangs, though `litecrawl` includes internal safeguards against stale locks.*

---

## Data Extraction

`litecrawl` handles the navigation and scheduling. You handle the data extraction via **hooks**.

```python
async def my_page_hook(page):
    """Click buttons or handle popups before parsing."""
    try:
        await page.click("#accept-cookies", timeout=1000)
    except:
        pass

def my_transform_hook(content: bytes, content_type: str, url: str) -> bytes:
    """Clean content before hashing (determines freshness)."""
    if "text/html" in content_type:
        # Return only the article body to ignore navigation/footer changes
        return extract_article_body(content) 
    return content

def my_downstream_hook(content: bytes, content_type: str, url: str, fresh: bool):
    """Save data to S3, Postgres, or JSON lines."""
    if fresh:
        save_to_s3(url, content)

litecrawl(
    ...,
    page_hook=my_page_hook,
    transform_hook=my_transform_hook,
    downstream_hook=my_downstream_hook
)
```

---

## Configuration

| Parameter | Purpose |
| :--- | :--- |
| `sqlite_path` | Path to the SQLite DB (auto-created). |
| `start_urls` | Seed URLs (inserted only if missing). |
| `normalize_patterns` | Regex replacements to standardize URLs (e.g., stripping session IDs). |
| `include/exclude_patterns` | Strict gating for which URLs are allowed in the frontier. |
| `n_claims` | Batch size. How many pages to lock for this specific run. |
| `n_concurrent` | Parallelism. How many browser tabs to open. |
| `pw_block_media` | Speed up crawling by aborting image/font requests (Default: `True`). |

---

## License

MIT
