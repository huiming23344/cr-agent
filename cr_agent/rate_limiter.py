from __future__ import annotations

import asyncio
import time
from typing import Any, Optional, Protocol


class RateLimiterProtocol(Protocol):
    async def __aenter__(self) -> None: ...
    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]: ...


class _AsyncLimiterBase:
    async def __aenter__(self):
        ...

    async def __aexit__(self, exc_type, exc, tb):
        return False


class AsyncRateLimiter(_AsyncLimiterBase):
    """简单限速器：确保任意两次调用间隔 >= 1 / qps 秒。"""

    def __init__(self, qps: float):
        self.interval = 1.0 / max(qps, 1e-6)
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def __aenter__(self):
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_time - now)
            if wait:
                await asyncio.sleep(wait)
            self._next_time = time.monotonic() + self.interval


class NoopRateLimiter(_AsyncLimiterBase):
    """No-op context manager used when no rate limit is configured."""


class RateLimitedLLM:
    """Wrap an LLM so every invoke/ainvoke acquires the limiter."""

    def __init__(self, llm: Any, limiter: RateLimiterProtocol):
        self._llm = llm
        self._limiter = limiter

    async def ainvoke(self, *args, **kwargs):
        async with self._limiter:
            return await self._llm.ainvoke(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("RateLimitedLLM.invoke cannot run inside an existing event loop")
        return asyncio.run(self.ainvoke(*args, **kwargs))

    def __getattr__(self, item):
        return getattr(self._llm, item)


__all__ = [
    "AsyncRateLimiter",
    "NoopRateLimiter",
    "RateLimiterProtocol",
    "RateLimitedLLM",
]
