import asyncio
import socket

import pytest

import litecrawl
from litecrawl import _is_safe_url


@pytest.fixture(autouse=True)
def clear_ssrf_cache():
    litecrawl._SSRF_SAFE_CACHE.clear()
    yield
    litecrawl._SSRF_SAFE_CACHE.clear()


@pytest.mark.asyncio
async def test_is_safe_url_blocks_private_ip_literal():
    assert await _is_safe_url("http://192.168.1.10/") is False


@pytest.mark.asyncio
async def test_is_safe_url_allows_public_ip_literal():
    assert await _is_safe_url("http://93.184.216.34/") is True


@pytest.mark.asyncio
async def test_is_safe_url_blocks_private_dns_result(monkeypatch):
    class LoopStub:
        async def getaddrinfo(self, _host, _port):
            return [(None, None, None, None, ("10.0.0.9", None))]

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: LoopStub())
    assert await _is_safe_url("http://example.test/") is False


@pytest.mark.asyncio
async def test_is_safe_url_caches_dns_failures(monkeypatch):
    class LoopStub:
        async def getaddrinfo(self, _host, _port):
            raise socket.gaierror("not resolvable")

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: LoopStub())
    url = "http://does-not-resolve/"
    assert await _is_safe_url(url) is False

    # Cached result should bypass resolution entirely
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: pytest.fail("DNS should be cached"))
    assert await _is_safe_url(url) is False
