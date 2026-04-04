import pydantic

from . import bind_enums

class PollParams(pydantic.BaseModel):
    keyboard_whitelist: frozenset[bind_enums.KeyBind] = pydantic.Field(
        default_factory=lambda: []
    )
    '''A set of all keyboard input features in the data.'''

    mouse_whitelist: frozenset[bind_enums.MouseBind] = pydantic.Field(
        default_factory=lambda: []
    )
    '''A set of all mouse input features in the data.'''

    gamepad_whitelist: frozenset[bind_enums.GamepadBind] = pydantic.Field(
        default_factory=lambda: []
    )
    '''A set of all gamepad input features in the data.'''

    capture_binds: frozenset[bind_enums.DigitalBind] = pydantic.Field(
        default_factory=lambda: [
            bind_enums.DigitalBind.LEFT_MOUSE, 
            bind_enums.DigitalBind.RIGHT_MOUSE
        ]
    )
    '''A set of binds that enable data capturing when held.'''

    capture_bind_gate: bind_enums.BindGate = pydantic.Field(
        default=bind_enums.BindGate.ANY
    )
    '''Whether any or all of the capture binds must be held to enable capturing.'''