"""
test_session_manager.py

Unit tests for services.session_manager (touch_activity, is_expired, reset_session, get/set_last_activity)
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from services.session_manager import SessionManager

class TestSessionManagerInit:
    """Tests for SessionManager __init__ (timeout_minutes)"""

    def test_default_timeout_30_minutes(self):
        sm = SessionManager()
        assert sm.timeout == timedelta(minutes=30)

    def test_custom_timeout(self):
        sm = SessionManager(timeout_minutes=5)
        assert sm.timeout == timedelta(minutes=5)

class TestSessionManagerIsExpired:
    """Tests for is_expired (no activity, just touched, after timeout)"""
    def test_no_activity_not_expired(self):
        """Session with no activity is not expired"""
        sm = SessionManager(timeout_minutes=30)
        assert sm._last_activity is None
        assert sm.is_expired() is False

    def test_just_touched_not_expired(self):
        """Session just touched is not expired"""
        sm = SessionManager(timeout_minutes=30)
        sm.touch_activity()
        assert sm.is_expired() is False

    def test_expired_after_timeout(self):
        """At exact timeout boundary (not greater), should not be expired"""
        sm = SessionManager(timeout_minutes=1)
        sm.touch_activity()
        past = sm._last_activity
        with patch("services.session_manager.datetime") as mock_dt:
            mock_dt.now.return_value = past + timedelta(minutes=2)
            assert sm.is_expired() is True

    def test_expired_after_timeout(self):
        """Session expired after timeout period"""
        sm = SessionManager(timeout_minutes=1)
        sm.touch_activity()
        past = sm._last_activity
        with patch("services.session_manager.datetime") as mock_dt:
            mock_dt.now.return_value = past + timedelta(minutes=2)
            assert sm.is_expired() is True

    def test_set_expired_timestamp_then_check(self):
        """Set past timestamp, then verify it's expired"""
        sm = SessionManager(timeout_minutes=30)
        past = datetime.now() - timedelta(minutes=60)
        sm.set_last_activity(past)
        assert sm.is_expired() is True

class TestSessionManagerTouchAndReset:
    """Tests for touch_activity, reset, get_last_activity, set_last_activity"""

    def test_touch_activity_sets_last_activity(self):
        """touch_activity sets last_activity timestamp"""
        sm = SessionManager()
        assert sm.get_last_activity() is None
        sm.touch_activity()
        assert sm.get_last_activity() is not None

    def test_multiple_touch_updates_last_activity(self):
        """Multiple touch_activity calls should update last_activity"""
        sm = SessionManager()
        sm.touch_activity()
        first = sm.get_last_activity()

        assert first is not None
        with patch("services.session_manager.datetime") as mock_dt:
            future = first + timedelta(seconds=5)
            mock_dt.now.return_value = future
            sm.touch_activity()
            second = sm.get_last_activity()
            assert second > first
            assert second == future

    def test_touch_after_expired_refreshes_session(self):
        """Touch activity after expiration should refresh the session"""
        sm = SessionManager(timeout_minutes=1)
        sm.touch_activity()
        past = sm._last_activity

        with patch("services.session_manager.datetime") as mock_dt:
            mock_dt.now.return_value = past + timedelta(minutes=2)
            assert sm.is_expired() is True
        sm.touch_activity()
        assert sm.is_expired() is False

    def test_reset_clears_last_activity(self):
        """reset clears last_activity to None"""
        sm = SessionManager()
        sm.touch_activity()
        sm.reset()
        assert sm.get_last_activity() is None
        assert sm.is_expired() is False

    def test_set_last_activity_restore(self):
        """set_last_activity restores timestamp"""
        sm = SessionManager()
        t = datetime(2026, 1, 2, 12, 0)
        sm.set_last_activity(t)
        assert sm.get_last_activity() == t

    def test_set_last_activity_to_none(self):
        """set_last_activity(None) should work like reset"""
        sm = SessionManager()
        sm.touch_activity()

        assert sm.get_last_activity() is not None
        sm.set_last_activity(None)
        assert sm.get_last_activity() is None
        assert sm.is_expired() is False