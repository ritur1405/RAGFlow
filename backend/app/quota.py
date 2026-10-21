"""Rate-limit handling for Gemini calls.

A 429 from Gemini means one of two things that need opposite handling:

    per-minute quota  -> wait for the window to reset, then retry
    per-day quota     -> nothing will succeed until tomorrow; fail now

Retrying everything blindly (as the SDK's own retry does for any status in its
list) turns an exhausted daily quota into many minutes of waiting before the
same error surfaces. with_quota_retry reads the error's QuotaFailure details,
fails immediately on a daily quota, and otherwise sleeps for the server's
suggested RetryInfo delay before retrying.
"""

from __future__ import annotations

import functools
import logging
import re
import time
from typing import Any, Callable, Optional

from google.genai import errors as genai_errors

logger = logging.getLogger("ragops.quota")

MAX_DELAY_SECONDS = 60.0


def _details(exc: genai_errors.APIError) -> list[dict]:
    body = exc.details if isinstance(exc.details, dict) else {}
    error = body.get("error", body)
    details = error.get("details") if isinstance(error, dict) else None
    return details if isinstance(details, list) else []


def is_daily_quota(exc: genai_errors.APIError) -> bool:
    """True when the 429 comes from a per-day quota, which retrying cannot fix."""
    for detail in _details(exc):
        for violation in detail.get("violations", []) or []:
            if "perday" in str(violation.get("quotaId", "")).lower():
                return True
    return False


def retry_delay_seconds(exc: genai_errors.APIError) -> Optional[float]:
    """The server's suggested wait from RetryInfo (e.g. "52s"), if present."""
    for detail in _details(exc):
        delay = detail.get("retryDelay")
        if isinstance(delay, str):
            match = re.fullmatch(r"\s*([\d.]+)s\s*", delay)
            if match:
                return float(match.group(1))
    return None


def with_quota_retry(fn: Callable[..., Any], attempts: int) -> Callable[..., Any]:
    """Wraps a blocking Gemini call so per-minute 429s are retried, daily ones are not."""

    @functools.wraps(fn)
    def call(*args, **kwargs):
        for attempt in range(1, attempts + 1):
            try:
                return fn(*args, **kwargs)
            except genai_errors.APIError as exc:
                if exc.code != 429 or attempt == attempts or is_daily_quota(exc):
                    raise
                delay = min(retry_delay_seconds(exc) or 2.0 ** attempt, MAX_DELAY_SECONDS)
                logger.warning(
                    "Gemini rate limit (attempt %d/%d); retrying in %.1fs", attempt, attempts, delay
                )
                time.sleep(delay)
        raise AssertionError("unreachable")  # the loop always returns or raises

    return call
