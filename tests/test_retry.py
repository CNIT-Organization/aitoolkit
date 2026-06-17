"""Unit tests for the async retry helper (``aitoolkit.retry.retry_async``)."""

from __future__ import annotations

import pytest

from aitoolkit.retry import retry_async


class Transient(Exception):
    """A retriable fault, for tests."""


@pytest.fixture
def no_sleep(monkeypatch):
    """Skip the backoff sleeps so retry tests run instantly."""

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("aitoolkit.retry.asyncio.sleep", _noop)


async def test_retries_transient_then_succeeds(no_sleep) -> None:
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Transient("flaky")
        return "ok"

    result = await retry_async(fn, attempts=3, retry_on=(Transient,))
    assert result == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on the third


async def test_non_retriable_surfaces_immediately(no_sleep) -> None:
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise ValueError("do not retry me")

    with pytest.raises(ValueError, match="do not retry me"):
        await retry_async(fn, attempts=3, retry_on=(Transient,))
    assert calls["n"] == 1  # not in retry_on -> tried exactly once


async def test_exhausts_attempts_and_reraises_last(no_sleep) -> None:
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise Transient(f"boom-{calls['n']}")

    with pytest.raises(Transient, match="boom-3"):
        await retry_async(fn, attempts=3, retry_on=(Transient,))
    assert calls["n"] == 3  # all attempts used, last error re-raised


async def test_succeeds_first_try_no_sleep(no_sleep) -> None:
    async def fn():
        return 42

    assert await retry_async(fn, attempts=3, retry_on=(Transient,)) == 42
