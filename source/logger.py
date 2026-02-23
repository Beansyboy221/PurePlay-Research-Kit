import logging
import time
import os
import constants

_logger = logging.getLogger("PurePlay")
logging.basicConfig(
    level=logging.INFO,
    format=constants.LOG_FORMAT
)

os.makedirs("logs", exist_ok=True)
_file_handler = logging.FileHandler(f'logs/{time.strftime(constants.TIMESTAMP_FORMAT)}.log')
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter(constants.LOG_FORMAT))
_logger.addHandler(_file_handler)

def info(msg, *args, **kwargs):
    _logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    _logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    _logger.error(msg, *args, **kwargs)

def exception(msg, *args, **kwargs):
    _logger.exception(msg, *args, **kwargs)

# DON'T FORGET TO LOG EXCEPTIONS