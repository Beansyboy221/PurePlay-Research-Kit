import pydantic

from data.polling import bind_enums

class PollParams(pydantic.BaseModel):
    polling_rate: int = pydantic.Field(default=60, gt=0)
    '''The polling rate used when polling the data (not the hardware polling rate).'''

    keyboard_whitelist: frozenset[bind_enums.KeyBind] = pydantic.Field(
        default=frozenset()
    )
    '''A set of all keyboard input features in the data.'''

    mouse_whitelist: frozenset[bind_enums.MouseBind] = pydantic.Field(
        default=frozenset()
    )
    '''A set of all mouse input features in the data.'''

    gamepad_whitelist: frozenset[bind_enums.GamepadBind] = pydantic.Field(
        default=frozenset()
    )
    '''A set of all gamepad input features in the data.'''

    capture_binds: frozenset[bind_enums.DigitalBind] = pydantic.Field(
        default=frozenset([
            bind_enums.DigitalBind.LEFT_MOUSE, 
            bind_enums.DigitalBind.RIGHT_MOUSE
        ])
    )
    '''A set of binds that enable data capturing when activated.'''

    capture_bind_gate: bind_enums.BindGate = pydantic.Field(
        default=bind_enums.BindGate.ANY
    )
    '''Whether any or all of the capture binds must be active to enable capturing.'''

    ignore_empty_polls: bool = pydantic.Field(default=True) # Shared between polling and processing params.
    '''Whether or not empty rows of features should be written to the data.'''

    reset_mouse_on_release: bool = pydantic.Field(default=True)
    '''Whether or not mouse deltas are reset to 0 when the capture bind is released.'''

    @property
    def whitelist(self) -> list[bind_enums.Bind]:
        '''All device whitelists combined.'''
        return self.keyboard_whitelist + self.mouse_whitelist + self.gamepad_whitelist
    
    @property
    def features_per_poll(self) -> int:
        '''The number of features in each poll (row).'''
        return len(self.keyboard_whitelist + self.mouse_whitelist + self.gamepad_whitelist)
