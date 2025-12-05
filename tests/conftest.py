import sys
import types


def _stub_aiosqlite() -> None:
    if "aiosqlite" in sys.modules:
        return

    async def connect_stub(*_args, **_kwargs):
        raise RuntimeError("aiosqlite stub used in tests; install dependencies to run crawler")

    class RowStub(dict):
        pass

    class ConnectionStub:
        pass

    sys.modules["aiosqlite"] = types.SimpleNamespace(
        Row=RowStub,
        Connection=ConnectionStub,
        connect=connect_stub,
    )


def _stub_lxml_html() -> None:
    if "lxml" in sys.modules:
        return

    class HTMLParserStub:
        def __init__(self, *_args, **_kwargs):
            pass

    class DocumentStub:
        def xpath(self, _expr):
            return []

    def fromstring_stub(_content, _parser=None):
        return DocumentStub()

    html_module = types.SimpleNamespace(HTMLParser=HTMLParserStub, fromstring=fromstring_stub)
    sys.modules["lxml"] = types.SimpleNamespace(html=html_module)
    sys.modules["lxml.html"] = html_module


def _stub_playwright_async_api() -> None:
    if "playwright.async_api" in sys.modules:
        return

    class BrowserContextStub:
        pass

    class PageStub:
        async def content(self):
            return ""

    class ResponseStub:
        def __init__(self):
            self.headers = {}

        async def body(self):
            return b""

    class RouteStub:
        request = types.SimpleNamespace(resource_type="document")

    class PlaywrightStub:
        def __init__(self):
            self.chromium = self

        async def launch(self, headless=True):
            return self

        async def new_context(self, **_kwargs):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

    async def async_playwright_stub():
        return PlaywrightStub()

    async_api_module = types.SimpleNamespace(
        BrowserContext=BrowserContextStub,
        Page=PageStub,
        Response=ResponseStub,
        Route=RouteStub,
        async_playwright=async_playwright_stub,
        TimeoutError=RuntimeError,
    )
    sys.modules["playwright.async_api"] = async_api_module
    sys.modules["playwright"] = types.SimpleNamespace(async_api=async_api_module)


_stub_aiosqlite()
_stub_lxml_html()
_stub_playwright_async_api()
