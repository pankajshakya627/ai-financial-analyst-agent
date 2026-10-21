"""API authentication and rate limiting."""

import hashlib
import secrets
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import Header, HTTPException, Request

from config.settings import settings

_WINDOW_SECONDS = 60.0
_MAX_TRACKED_CLIENTS = 10_000

_hits: dict[str, deque] = defaultdict(deque)


def client_id(request: Request, api_key: Optional[str]) -> str:
    """Stable identifier for rate limiting and session namespacing."""
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


async def authenticate(
    request: Request, x_api_key: Optional[str] = Header(default=None)
) -> None:
    """Require the configured API key (if any), then apply the rate limit."""
    expected = settings.api_key
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    cid = client_id(request, x_api_key if expected else None)
    request.state.client_id = cid

    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return

    now = time.monotonic()
    window = _hits[cid]
    while window and now - window[0] > _WINDOW_SECONDS:
        window.popleft()
    if len(window) >= limit:
        retry_after = max(1, int(_WINDOW_SECONDS - (now - window[0])))
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
    window.append(now)

    if len(_hits) > _MAX_TRACKED_CLIENTS:  # drop idle clients
        for key in [k for k, v in _hits.items() if not v or now - v[-1] > _WINDOW_SECONDS]:
            _hits.pop(key, None)


def reset_rate_limits() -> None:
    """Clear rate-limit state (used by tests)."""
    _hits.clear()
