import tkinter.ttk

from source.prompters import base_prompter

class IntPrompter(base_prompter.BasePrompter):
    output_type = int

    def prompt_with_cli(self, title: str, description: str) -> int:
        while True:
            val = input(f'\n{title}\n{description} (Integer)\n> ').strip()
            try:
                return int(val)
            except:
                raise ValueError('Invalid integer.')

    def prompt_with_tk(self, title: str, description: str) -> int:
        raise NotImplementedError