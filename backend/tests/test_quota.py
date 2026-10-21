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
            asyncio.run(eg._call_gemini_judge("sys", "prompt", "faithfulness"))
    assert client.models.generate_content.call_count == 1
