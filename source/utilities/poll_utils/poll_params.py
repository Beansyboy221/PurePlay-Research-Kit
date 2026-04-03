import pydantic

from . import bind_enums

class PollParams(pydantic.BaseModel): # How should this relate to data_params?
    keyboard_whitelist: frozenset[bind_enums.KeyBind] = pydantic.Field(
        default_factory=lambda: [],
        description='A set of all keyboard input features in the data.',
        validation_alias=pydantic.AliasPath('collect', 'keyboard_whitelist')
    )
    '''A set of all keyboard input features in the data.'''

    mouse_whitelist: frozenset[bind_enums.MouseBind] = pydantic.Field(
        default_factory=lambda: [],
        description='A set of all mouse input features in the data.',
        validation_alias=pydantic.AliasPath('collect', 'mouse_whitelist')
    )
    '''A set of all mouse input features in the data.'''

    gamepad_whitelist: frozenset[bind_enums.GamepadBind] = pydantic.Field(
        default_factory=lambda: [],
        description='A set of all gamepad input features in the data.',
        validation_alias=pydantic.AliasPath('collect', 'gamepad_whitelist')
    )
    '''A set of all gamepad input features in the data.'''