import importlib
import pkgutil

# Recursively import all readers
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, __name__ + '.'):
    importlib.import_module(module_name)