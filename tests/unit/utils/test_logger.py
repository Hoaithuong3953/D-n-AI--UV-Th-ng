"""
test_logger.py

Unit tests for setup_logger with mocked settings (no real .env)
"""
import logging
import logging.handlers
from unittest.mock import MagicMock, patch

import pytest

@pytest.fixture
def mock_log_settings():
    m = MagicMock()
    m.LOG_LEVEL = "INFO"
    m.LOG_FORMAT = "%(message)s"
    m.LOG_DATE_FORMAT = "%Y-%m-%d"
    m.LOG_TO_FILE = False
    return m

def test_setup_logger_adds_console_handler(mock_log_settings):
    from utils.logger import setup_logger

    name = "learnpath_unit_test_logger_console"
    with patch("utils.logger.settings", mock_log_settings):
        log = setup_logger(name)

    try:
        assert log.propagate is False
        assert log.level == logging.INFO
        assert len(log.handlers) == 1
        assert isinstance(log.handlers[0], logging.StreamHandler)
    finally:
        lg = logging.getLogger(name)
        lg.handlers.clear()

def test_setup_logger_adds_file_handler_when_enabled(tmp_path, mock_log_settings):
    from utils.logger import setup_logger

    log_file = tmp_path / "app.log"
    mock_log_settings.LOG_TO_FILE = True
    mock_log_settings.LOG_FILE_PATH = str(log_file)
    mock_log_settings.LOG_FILE_ROTATION = "midnight"
    mock_log_settings.LOG_FILE_RETENTION = 7

    name = "learnpath_unit_test_logger_file"
    with patch("utils.logger.settings", mock_log_settings):
        log = setup_logger(name)

    try:
        assert len(log.handlers) == 2
        kinds = {type(h) for h in log.handlers}
        assert logging.StreamHandler in kinds
        assert logging.handlers.TimedRotatingFileHandler in kinds
        assert log_file.parent.exists()
    finally:
        lg = logging.getLogger(name)
        lg.handlers.clear()