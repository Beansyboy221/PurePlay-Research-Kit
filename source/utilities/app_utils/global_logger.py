import logging
import enum
import time
import os

from globals import formats

#region Enums
class LogLevel(enum.StrEnum):
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'
    FATAL = 'FATAL'
#endregion

#region Functions
def set_log_level(log_level: LogLevel) -> None:
    '''Changes the log level of the logger and file handler.'''
    _logger.setLevel(log_level)
    _file_handler.setLevel(log_level)

def debug(msg, *args, **kwargs): 
    '''Logs a message with severity 'DEBUG'.'''
    _logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    '''Logs a message with severity 'INFO'.'''
    _logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    '''Logs a message with severity 'WARNING'.'''
    _logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    '''Logs a message with severity 'ERROR'.'''
    _logger.error(msg, *args, **kwargs)

def exception(msg, *args, **kwargs): # I recommend to just throw one of these as a handler of exceptions in each of your entry points.
    '''Logs a message with severity 'ERROR' and an exception traceback.'''
    _logger.exception(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    '''Logs a message with severity 'CRITICAL'.'''
    _logger.critical(msg, *args, **kwargs)
#endregion

#region Setup
_logger = logging.getLogger('PurePlay')
logging.basicConfig(
    level=logging.INFO,
    format=formats.LOG_FORMAT
	)

os.makedirs('logs', exist_ok=True)
_file_handler = logging.FileHandler(f'logs/{time.strftime(formats.TIMESTAMP_FORMAT)}.log')
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter(formats.LOG_FORMAT))
_logger.addHandler(_file_handler)
#endregion