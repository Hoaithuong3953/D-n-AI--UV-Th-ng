"""
session_manager.py

Session manager for tracking user activity and session expiration

Key features:
- Track last activity timestamp; check if session expired due to inactivity
- touch_activity, is_expired, reset for lightweight in-memory session
"""
from datetime import datetime, timedelta

class SessionManager:
    """
    Manage session lifetime based on user activity (in-memory)

    Responsibilities:
    - touch_activity: record current time as last activity
    - is_expired: true when inactivity exceeds timeout
    - reset: clear last activity (e.g. new session)
    """
    def __init__(self, timeout_minutes: int = 30):
        """
        Initialize a new session manager with specified inactivity timeout

        Args:
            timeout_minutes: Number of minutes of inactivity before expiration
                Default: 30 mins
        """
        self.timeout = timedelta(minutes=timeout_minutes)
        self._last_activity: datetime | None = None

    def touch_activity(self) -> None:
        """Record current time as the last user activity"""
        self._last_activity = datetime.now()

    def is_expired(self) -> bool:
        """Return True if the session has expired due to inactivity (no activity for timeout period)"""
        if self._last_activity is None:
            return False
        return datetime.now() - self._last_activity > self.timeout

    def reset(self) -> None:
        """Reset the session to its initial state"""
        self._last_activity = None

    def get_last_activity(self):
        """Return the last activity datetime (for session persistence) or None"""
        return self._last_activity
    
    def set_last_activity(self, value: datetime | None) -> None:
        """Restore the last activity (e.g. from session_state)"""
        self._last_activity = value