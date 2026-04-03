from . import base_mode

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

def register_prompter(prompter_class: type[base_mode.ProgramMode]) -> None:
    AVAILABLE_MODES[prompter_class.name] = prompter_class
base_mode.ProgramMode.on_init.connect(register_prompter)