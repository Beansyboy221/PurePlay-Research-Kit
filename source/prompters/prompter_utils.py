from . import base_prompter

# Considerations:
# Ensure .get results in type being highlighted
# The order in which this is imported matters
# Should this be defined when it is needed or at the start of the program?

AVAILABLE_PROMPTERS = dict[type, type[base_prompter.BasePrompter]]
'''
A registry of all loaded prompters.
Use this to dynamically find prompters.
'''

def register_prompter(prompter_class: type[base_prompter.BasePrompter]) -> None:
    AVAILABLE_PROMPTERS[prompter_class.output_type] = prompter_class
base_prompter.BasePrompter.on_init.connect(register_prompter)