import importlib
import pkgutil

from modes import base_mode

# Considerations:
# Ensure .get results in type being highlighted
# The order in which this is imported matters
# Should this be defined when it is needed or at the start of the program?
AVAILABLE_MODES = dict[str, type[base_mode.ProgramMode]]
'''
A registry of all loaded modes.
Use this to dynamically find modes.
Key: the mode's name
Value: the mode class
'''

def register_mode(mode: type[base_mode.ProgramMode]) -> None:
    AVAILABLE_MODES[mode.name] = mode
base_mode.ProgramMode.on_init_subclass.connect(register_mode)

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, __name__ + '.'):
    importlib.import_module(module_name)