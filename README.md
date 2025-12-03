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

Requires Python 3.8+ and Playwright.

```bash
pip install aiosqlite playwright lxml
playwright install chromium
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
* * * * * /usr/bin/timeout 10m /usr/bin/python3 /path/to/crawler.py >> /var/log/crawl.log 2>&1
```

## Core Features

### 1. Robust Normalization & Discovery
URLs are the primary key. `litecrawl` automatically standardizes URLs (sorting query parameters, stripping fragments) to prevent duplication. It creates a directed graph where `last_inlink_seen_time` helps prune orphaned pages that are no longer linked by the target site.

### 2. Security (SSRF Protection)
When configured with `check_ssrf=True` (default), the crawler performs DNS resolution to ensure it does not connect to private IP ranges (e.g., `127.0.0.1`, `10.0.0.0/8`). This is essential when running crawlers on infrastructure that has access to internal services.

### 3. Change Detection
It computes a stable SHA-256 hash of the page content.
*   **Fresh:** If the hash changes or new links are discovered, the page is marked "fresh," and the interval to the next crawl is reduced.
*   **Stale:** If the hash remains identical, the interval is increased (backed off) to save resources.

### 4. Hook System
You can inject custom logic without modifying the core library:

*   **`page_ready_hook`**: Run actions after the page loads but before extraction (e.g., clicking "Load More", logging in, handling cookie banners).
*   **`link_extract_hook`**: Override default link discovery (e.g., extracting URLs from JSON blobs).
*   **`content_extract_hook`**: Define exactly what data to save (e.g., returning a specific DOM element or a structured dict).
*   **`downstream_hook`**: An async callback triggered immediately after a successful crawl. Use this to push data to an API, Kafka, or S3.

## Use Case Examples

### The "News Monitor"
Targeting a high-frequency news feed.
*   **Strategy:** Aggressive `fresh_factor`, specific inclusion patterns.
*   **Config:**
    ```python
    litecrawl(
        ...,
        fresh_factor=0.2,   # Check 5x more often if it changed
        min_interval_sec=300,
        include_patterns=[r"/breaking-news/"]
    )
    ```

### The "Intranet Archivist"
Crawling internal documentation for search indexing.
*   **Strategy:** Disable SSRF checks (to allow internal IPs), long intervals, authentication via hooks.
*   **Config:**
    ```python
    async def login(page):
        # Custom login logic
        ...

    litecrawl(
        ...,
        check_ssrf=False,
        page_ready_hook=login,
        new_interval_sec=86400 * 7, # Default to weekly
        stale_factor=1.5
    )
    ```

## Operational Best Practices

1.  **Concurrency:** Keep `n_concurrent` moderate (5-20). SQLite handles concurrency well, but high write contention from hundreds of workers on a single file can lead to locking issues.
2.  **Database Management:** The `sqlite_path` is the only state. Backup this file to back up your crawl frontier.
3.  **Logs:** The tool logs to standard python logging. Redirect stdout/stderr to a file in your cron definition to monitor progress.
4.  **Partitioning:** If you need to crawl 10 distinct websites, it is often better to set up 10 separate cron entries with 10 separate SQLite files rather than one massive monolithic crawl. This isolates failures and simplifies configuration.