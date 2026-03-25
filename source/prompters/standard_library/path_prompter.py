import tkinter.ttk
import pathlib

from source.prompters import base_prompter

class PathPrompter(base_prompter.BasePrompter):
    output_type = pathlib.Path

    def prompt_with_cli(
            self, 
            title: str, 
            description: str
        ) -> pathlib.Path:
        raise NotImplementedError

    def prompt_with_tk(
            self, 
            title: str, 
            description: str
        ) -> pathlib.Path:
        raise NotImplementedError