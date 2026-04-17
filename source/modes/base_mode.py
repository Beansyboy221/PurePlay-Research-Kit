import pydantic
import typing

from source.misc import dunder_signals

class ProgramMode(pydantic.BaseModel, dunder_signals.OnInitSubclassMixin):
    '''Base class for all program modes.'''
    name: str
    '''The name of the mode.'''
    
    description: str
    '''A description of the mode.'''

    entry_point: typing.Callable
    '''The entry point of the mode.'''

    config: type[pydantic.BaseModel]
    '''The config class needed for the mode.'''