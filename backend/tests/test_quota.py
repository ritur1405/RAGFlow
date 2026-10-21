"""Tests for the quota-aware 429 handling in app/quota.py."""

from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from app import quota


def _429(quota_id="GenerateRequestsPerMinutePerProjectPerModel-FreeTier", retry_delay="7s"):
    details = [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": quota_id, "quotaValue": "20"}]},
    ]
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    return genai_errors.ClientError(
        429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED", "details": details}}
    )


DAILY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def test_parses_daily_quota_and_retry_delay():
    assert quota.is_daily_quota(_429(DAILY)) is True
    assert quota.is_daily_quota(_429()) is False
    assert quota.retry_delay_seconds(_429(retry_delay="52.5s")) == 52.5
    assert quota.retry_delay_seconds(_429(retry_delay=None)) is None


def test_per_minute_429_waits_for_the_server_delay_then_succeeds():
    fn = MagicMock(side_effect=[_429(retry_delay="7s"), "ok"])
    with patch.object(quota.time, "sleep") as sleep:
        assert quota.with_quota_retry(fn, attempts=3)("a", k=1) == "ok"
    sleep.assert_called_once_with(7.0)
    assert fn.call_count == 2
    fn.assert_called_with("a", k=1)


def test_delay_is_capped():
    fn = MagicMock(side_effect=[_429(retry_delay="600s"), "ok"])
    with patch.object(quota.time, "sleep") as sleep:
        quota.with_quota_retry(fn, attempts=3)()
    sleep.assert_called_once_with(quota.MAX_DELAY_SECONDS)


def test_daily_quota_fails_immediately():
    fn = MagicMock(side_effect=_429(DAILY))
    with patch.object(quota.time, "sleep") as sleep:
        with pytest.raises(genai_errors.ClientError):
            quota.with_quota_retry(fn, attempts=8)()
    assert fn.call_count == 1
    sleep.assert_not_called()


def test_gives_up_after_the_last_attempt():
    fn = MagicMock(side_effect=_429())
    with patch.object(quota.time, "sleep"):
        with pytest.raises(genai_errors.ClientError):
            quota.with_quota_retry(fn, attempts=3)()
    assert fn.call_count == 3


def test_other_errors_are_not_retried_here():
    err = genai_errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID"}})
    fn = MagicMock(side_effect=err)
    with pytest.raises(genai_errors.ClientError):
        quota.with_quota_retry(fn, attempts=5)()
    assert fn.call_count == 1


def test_judge_reports_a_daily_quota_as_unavailable_without_retrying():
    import asyncio

    from app import eval_generation as eg

    client = MagicMock()
    client.models.generate_content.side_effect = _429(DAILY)
    with patch.object(eg, "_client", client):
        with pytest.raises(eg.JudgeUnavailableError, match="daily quota"):
            asyncio.run(eg.run_generation_eval("Q?", "A", [eg.ContextChunk("1", "t")]))
    assert client.models.generate_content.call_count == 1


def test_find_rate_limit_follows_wrapped_causes():
    limit = _429(DAILY)
    try:
        try:
            raise limit
        except genai_errors.ClientError as inner:
            raise RuntimeError("LLM generation failed") from inner
    except RuntimeError as outer:
        assert quota.find_rate_limit(outer) is limit
    assert quota.find_rate_limit(RuntimeError("unrelated")) is None


def test_rate_limit_messages_name_the_model_and_the_kind_of_limit():
    daily = genai_errors.ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "Quota exceeded for metric: x, limit: 500, model: gemini-3.5-flash-lite",
        "details": [{"violations": [{"quotaId": DAILY}]}]}})
    assert "daily request limit for gemini-3.5-flash-lite" in quota.rate_limit_message(daily)
    assert "Try again in a minute" in quota.rate_limit_message(_429())


def test_query_endpoint_reports_a_quota_as_503():
    from fastapi.testclient import TestClient

    from app.database import get_db
    from app.main import app

    app.dependency_overrides[get_db] = lambda: MagicMock()
    try:
        with patch("app.main.query_rag", side_effect=_429(DAILY)), TestClient(app) as client:
            resp = client.post("/query/", json={"question": "What is BM25?"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 503
    assert "daily request limit" in resp.json()["detail"]
