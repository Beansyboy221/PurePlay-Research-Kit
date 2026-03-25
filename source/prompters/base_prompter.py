import abc

from utilities.mixins import on_init_mixin

class BasePrompter(abc.ABC, on_init_mixin.OnInitMixin):
    '''Base class is simply for enforcing mixins.'''
    @property
    @abc.abstractmethod
    def output_type(self) -> type:
        raise NotImplementedError
    
    def prompt_with_cli(
            self, 
            title: str, 
            description: str
        ) -> output_type:
        '''Prompts the user using the CLI.'''
        ...

    def prompt_with_gui(
            self, 
            title: str, 
            description: str
        ) -> output_type:
        '''Prompts the user using a GUI.'''
        ...

    # If I make these strategies, I can have more.
    # For example:
    # prompt_with_SPECIFIC_GUI_MODULE
    # or
    # prompt_with_NPC