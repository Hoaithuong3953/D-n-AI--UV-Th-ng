"""
Session manager for tracking user activity and session expiration

This module provides a lightweight, in-memory SessionManager class to:
- Track the last user activity timestamp
- Determine if a session has timed out due to inactivity
- Support simple reset functionality
"""
from datetime import datetime, timedelta

class SessionManager:
    """
    Manage session lifetime based on user activity
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
        """Check if the current session has expired based on inactivity
        
        Returns:
            True if no activity has occurred for longer than the timeout period
        """
        if self._last_activity is None:
            return False
        return datetime.now() - self._last_activity > self.timeout

    def reset(self) -> None:
        """Reset the session to its initial state"""
        self._last_activity = None