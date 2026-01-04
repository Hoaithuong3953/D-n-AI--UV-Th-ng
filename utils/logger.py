import logging
import sys
from config.settings import settings

def setup_logger(name: str = "LearnPath"):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(settings.LOG_LEVEL)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()