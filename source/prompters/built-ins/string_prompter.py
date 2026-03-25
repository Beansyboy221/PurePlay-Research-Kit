import tkinter.ttk

from source.prompters import base_prompter

class StringPrompter(base_prompter.BasePrompter):
    output_type = str

    def prompt_with_cli(self, title: str, description: str) -> str:
        return input(f'\n{title}\n{description}\n> ').strip()

    def prompt_with_tk(self, title: str, description: str) -> str:
        raise NotImplementedError