import logging
from pathlib import Path
from lxml import html
from litecrawl import litecrawl


def retain_title_only(content: bytes, content_type: str, url: str) -> bytes:
    """Transform hook that keeps only the <title> text for HTML pages."""
    if not content_type.lower().startswith("text/html"):
        return content
    try:
        document = html.fromstring(content)
        title = (document.findtext(".//title") or "").strip()
        return title.encode("utf-8")
    except Exception:  # noqa: BLE001
        return content


def log_title(content: bytes, content_type: str, url: str, fresh: bool) -> None:
    """Downstream hook that logs the extracted title to console."""
    if not content_type.lower().startswith("text/html"):
        return
    title = content.decode("utf-8", errors="ignore").strip()
    logging.info("Page title [%s]: %s", url, title or "<empty>")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    db_path = Path(__file__).with_name("litecrawl.db")
    logging.info("Starting litecrawl run; db=%s", db_path)
    litecrawl(
        sqlite_path=str(db_path),
        start_urls=["https://aarhus.dk/"],
        include_patterns=[r"^https?://([^.]+\.)*aarhus\.dk/"],
        exclude_patterns=[],
        transform_hook=retain_title_only,
        downstream_hook=log_title,
    )
    logging.info("litecrawl run completed")


if __name__ == "__main__":
    main()