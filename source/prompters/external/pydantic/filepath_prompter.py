import tkinter.ttk
import pydantic

from source.prompters import base_prompter

class FilePathPrompter(base_prompter.BasePrompter):
    output_type = pydantic.FilePath

    def prompt_with_cli(
            self, 
            title: str, 
            description: str
        ) -> pydantic.FilePath:
        raise NotImplementedError

    def prompt_with_tk(
            self, 
            title: str, 
            description: str
        ) -> pydantic.FilePath:
        raise NotImplementedError