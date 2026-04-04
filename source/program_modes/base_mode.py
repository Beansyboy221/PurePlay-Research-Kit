import pydantic
import typing

from utilities.mixins import on_init_mixin

class ProgramMode(pydantic.BaseModel, on_init_mixin.OnInitMixin):
    '''Base class for all program modes.'''
    name: str = pydantic.Field(
        default_factory=lambda: ProgramMode.__name__.lower(),
        const=True
	)
    '''The name of the mode.'''
    
    description: str = pydantic.Field(
        default='A description of the mode.',
        const=True
	)
    '''A description of the mode.'''

    entry_point: typing.Callable = pydantic.Field(const=True)
    '''The entry point of the mode.'''

    config: type[pydantic.BaseModel]
    '''The config class needed for the mode.'''