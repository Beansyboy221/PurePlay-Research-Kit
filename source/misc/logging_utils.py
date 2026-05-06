"""A module containing extended utilities for logging."""

import logging
import enum
import time
import sys
import os

from globals import formats

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIRNAME = "logs"
LOG_FILENAME = f"{time.strftime(formats.TIMESTAMP_FORMAT)}.log"


class LogLevel(enum.StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"


os.makedirs(LOG_DIRNAME, exist_ok=True)


def get_logger(name: str = __name__) -> logging.Logger:
    """Gets/creates a logger with a given name."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    logger.setLevel(LogLevel.DEBUG)

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=formats.TIMESTAMP_FORMAT,
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LogLevel.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(f"{LOG_DIRNAME}/{LOG_FILENAME}")
    file_handler.setLevel(LogLevel.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
