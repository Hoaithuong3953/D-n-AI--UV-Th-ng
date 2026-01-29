"""
Logging configuration helpers and shared logger instance
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from config import settings

def setup_logger(name: str = "learnpath-bot") -> logging.Logger:
    """
    Configure and return a named logger with console and optinal file output

    Configuration is loaded from `settings`

    Args:
        name: Logger name defaults to 'learnpath-bot'

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

        # Use custom format if provided
        default_format = "%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s"
        formatter = logging.Formatter(
            fmt=settings.LOG_FORMAT or default_format,
            datefmt=settings.LOG_DATE_FORMAT
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if settings.LOG_TO_FILE:
            log_path = Path(settings.LOG_FILE_PATH)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handlers = logging.handlers.TimedRotatingFileHandler(
                filename=log_path,
                when=settings.LOG_FILE_ROTATION,
                interval=1,
                backupCount=settings.LOG_FILE_RETENTION,
                encoding="utf-8"
            )

            file_handlers.setFormatter(formatter)
            logger.addHandler(file_handlers)

    logger.propagate = False
    return logger

logger = setup_logger()