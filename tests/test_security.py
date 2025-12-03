import pytest

from litecrawl import _is_safe_url


@pytest.mark.asyncio
async def test_is_safe_url_blocks_private_ip_literal():
    assert await _is_safe_url("http://192.168.1.10/") is False


@pytest.mark.asyncio
async def test_is_safe_url_allows_public_ip_literal():
    assert await _is_safe_url("http://93.184.216.34/") is True
