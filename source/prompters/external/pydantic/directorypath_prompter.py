import tkinter.ttk
import pydantic

from source.prompters import base_prompter

class PathPrompter(base_prompter.BasePrompter):
    output_type = pydantic.DirectoryPath

    def prompt_with_cli(
            self, 
            title: str, 
            description: str
        ) -> pydantic.DirectoryPath:
        raise NotImplementedError

    def prompt_with_tk(
            self, 
            title: str, 
            description: str
        ) -> pydantic.DirectoryPath:
        raise NotImplementedError