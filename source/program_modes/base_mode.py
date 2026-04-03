import pydantic
import typing

from utilities.mixins import on_init_mixin

class ProgramMode(pydantic.BaseModel, on_init_mixin.OnInitMixin):
    '''Base class for all program modes.'''
    name: str = pydantic.Field(
        default_factory=lambda: ProgramMode.__name__.lower(),
        const=True,
        description='The name of the mode.'
    )
    '''The name of the mode.'''
    
    description: str = pydantic.Field(
        default='A description of the mode.',
        const=True,
        description='A description of the mode.'
    )
    '''A description of the mode.'''

    entry_point: typing.Callable = pydantic.Field(
        const=True,
        description='The entry point of the mode.'
    )
    '''The entry point of the mode.'''

    config: type[pydantic.BaseModel] = pydantic.Field(
        description='The config class needed for the mode.'
    )
    '''The config class needed for the mode.'''