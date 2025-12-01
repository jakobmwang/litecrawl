"""Minimal runnable entry point for cron usage."""

from litecrawl import litecrawl


if __name__ == "__main__":
    litecrawl(
        sqlite_path="litecrawl.db",
        start_urls=["https://example.com/"],
        normalize_patterns=[{"pattern": r"/+$", "replace": ""}],
        include_patterns=[r"example\\.com"],
        exclude_patterns=[],
    )
