"""Shared errors for external model-provider boundaries."""

import math
import re
import time
from collections.abc import Iterator


class ProviderRateLimitError(RuntimeError):
    """An external provider rejected a request because of quota or rate limits."""

    def __init__(
        self,
        provider: str,
        *,
        retry_after_seconds: int | None = None,
    ):
        self.provider = provider
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"{provider} provider rate limit exceeded")


def is_provider_rate_limit_error(exc: BaseException) -> bool:
    """Recognize quota errors without depending on one SDK's exception type."""
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status == 429 or str(status) == "429":
        return True

    details = " ".join(str(item) for item in _exception_chain(exc)).lower()
    return any(
        marker in details
        for marker in ("resource_exhausted", "rate limit", "rate_limit", "quota")
    )


def provider_rate_limit_error(
    provider: str, exc: BaseException
) -> ProviderRateLimitError:
    """Translate an SDK exception into the application's stable error type."""
    return ProviderRateLimitError(
        provider,
        retry_after_seconds=_retry_after_seconds(exc),
    )


def provider_retry_after_seconds(exc: BaseException) -> int | None:
    """Return a provider-supplied retry delay when one is available."""
    return _retry_after_seconds(exc)


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _retry_after_seconds(exc: BaseException) -> int | None:
    for current in _exception_chain(exc):
        headers = getattr(current, "headers", None)
        if headers is None:
            headers = getattr(getattr(current, "response", None), "headers", None)
        if not headers:
            continue
        retry_after_ms = _header_value(headers, "retry-after-ms")
        if retry_after_ms is not None:
            try:
                return max(1, math.ceil(float(retry_after_ms) / 1000))
            except (TypeError, ValueError):
                pass
        retry_after = _header_value(headers, "retry-after")
        if retry_after is not None:
            try:
                return max(1, math.ceil(float(retry_after)))
            except (TypeError, ValueError):
                pass
        reset = _header_value(headers, "x-ratelimit-reset")
        if reset is not None:
            try:
                reset_value = float(reset)
                delay = (
                    reset_value - time.time()
                    if reset_value > time.time()
                    else reset_value
                )
                return max(1, math.ceil(delay))
            except (TypeError, ValueError):
                pass
    details = " ".join(str(item) for item in _exception_chain(exc))
    match = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", details, re.IGNORECASE)
    if match is None:
        return None
    return max(1, math.ceil(float(match.group(1))))


def _header_value(headers, name: str):
    for key, value in headers.items():
        if str(key).lower() == name:
            return value
    return None
