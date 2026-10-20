"""API-key protection for endpoints that cost money to call.

This project had no authentication of any kind, so there was nothing to reuse.
Rather than pull in a user/session/JWT stack that nothing else here needs, this
is a shared-secret header check built on FastAPI's own APIKeyHeader -- no new
dependencies, and configured through an environment variable like every other
setting in the project.

Scope is deliberately narrow: it guards *execution* (POST /api/eval/run), which
spends Gemini credits on every call. The read-only endpoints stay open so the
dashboard can display results without shipping a secret to the browser.

Fail-closed by design: if EVAL_API_KEY is unset, the guarded endpoint returns
503 rather than allowing the call. An unconfigured deployment therefore exposes
nothing, which is the whole point of adding this. Set EVAL_API_KEY to enable it.

    EVAL_API_KEY=<a long random string>

and send it on guarded requests as:

    X-API-Key: <that string>
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger("ragops.auth")

API_KEY_HEADER_NAME = "X-API-Key"

# auto_error=False so a missing header reaches our handler and produces a
# consistent message instead of FastAPI's default 403.
_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def configured_eval_api_key() -> str | None:
    """Reads EVAL_API_KEY at call time so configuration can change per process."""
    key = os.environ.get("EVAL_API_KEY", "").strip()
    return key or None


def eval_auth_configured() -> bool:
    return configured_eval_api_key() is not None


async def require_eval_api_key(
    provided: str | None = Depends(_api_key_header),
) -> None:
    """Authorizes a request to run an evaluation.

    Raises 503 when no key is configured, 401 when the supplied key is absent
    or wrong. Comparison is constant-time so a valid key cannot be recovered by
    timing repeated requests.
    """
    expected = configured_eval_api_key()

    if expected is None:
        logger.error(
            "Rejected a request to a guarded endpoint: EVAL_API_KEY is not set, "
            "so the endpoint is closed."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "This endpoint is disabled because EVAL_API_KEY is not configured "
                "on the server. Set EVAL_API_KEY to enable it."
            ),
        )

    if not provided or not secrets.compare_digest(provided, expected):
        logger.warning(
            "Rejected a request to a guarded endpoint: %s",
            "missing API key" if not provided else "invalid API key",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"A valid {API_KEY_HEADER_NAME} header is required.",
            headers={"WWW-Authenticate": API_KEY_HEADER_NAME},
        )
