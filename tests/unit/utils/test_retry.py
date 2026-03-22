"""
test_retry.py

Unit tests for utils.retry: TRANSIENT_ERRORS, gemini_retry behavior
"""
import pytest
from unittest.mock import patch
from google.api_core import exceptions as google_exceptions
from tenacity import wait_none

from utils.retry import TRANSIENT_ERRORS, gemini_retry

def _noop_before_sleep_log(*args, **kwargs):
    """Tenacity before_sleep callback; avoids before_sleep_log + Logger incompatibility in tests"""
    return lambda retry_state: None

def test_transient_errors_contains_expected_google_exceptions():
    expected = (
        google_exceptions.DeadlineExceeded,
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.Aborted,
    )
    assert TRANSIENT_ERRORS == expected

def test_gemini_retry_succeeds_after_transient_failure():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise google_exceptions.ServiceUnavailable("retry me")
        return "ok"

    with patch("utils.retry.wait_exponential", return_value=wait_none()), patch(
        "utils.retry.before_sleep_log",
        side_effect=_noop_before_sleep_log,
    ):
        wrapped = gemini_retry(max_retries=3)(flaky)
        assert wrapped() == "ok"
    assert calls["n"] == 2

def test_gemini_retry_does_not_retry_non_transient_error():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("not transient")

    with patch("utils.retry.wait_exponential", return_value=wait_none()), patch(
        "utils.retry.before_sleep_log",
        side_effect=_noop_before_sleep_log,
    ):
        wrapped = gemini_retry(max_retries=3)(boom)
        with pytest.raises(ValueError, match="not transient"):
            wrapped()
    assert calls["n"] == 1

def test_gemini_retry_reraises_after_exhausting_attempts():
    def always_unavailable():
        raise google_exceptions.ServiceUnavailable("down")

    with patch("utils.retry.wait_exponential", return_value=wait_none()), patch(
        "utils.retry.before_sleep_log",
        side_effect=_noop_before_sleep_log,
    ):
        wrapped = gemini_retry(max_retries=2)(always_unavailable)
        with pytest.raises(google_exceptions.ServiceUnavailable, match="down"):
            wrapped()
