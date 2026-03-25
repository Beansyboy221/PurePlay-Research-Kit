import tkinter.ttk

from globals.constants import formats
from utilities.app_utils import global_logger
from source.prompters import base_prompter

class BoolPrompter(base_prompter.BasePrompter):
    output_type = bool

    def prompt_with_cli(self, title: str, description: str) -> bool:
        while True:
            value = input(f'\n{title} (y/n)\n{description}\n').strip()
            bool = formats.BOOL_STRINGS.get(value)
            if bool is None:
                global_logger.warning('Invalid boolean value. Please try again.')
            return bool

    def prompt_with_tk(self, title: str, description: str) -> bool:
        raise NotImplementedError