'''A framework for building ML models for PurePlay'''
import importlib
import pkgutil

from misc import logging_utils
from . import helpers # Import first to ensure modules register correctly

logger = logging_utils.get_logger()

logger.info('Searching for models to register...')
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, __name__ + '.'):
    importlib.import_module(module_name)