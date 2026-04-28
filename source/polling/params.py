import pydantic

from misc import validated_types
from polling.controller import binds as controller_binds
from polling.keyboard import binds as key_binds
from polling.mouse import binds as mouse_binds
from . import (
    unified_binds,
    base_bind
)

class PollParams(pydantic.BaseModel):
    '''Parameters for polling data.'''
    polling_rate: int = pydantic.Field(default=60, gt=0)
    '''The polling rate used when polling the data (not the hardware polling rate).'''

    keyboard_whitelist: frozenset[key_binds.Bind] = pydantic.Field(
        default=frozenset()
    )
    '''A set of keyboard input features to poll for.'''

    mouse_whitelist: frozenset[mouse_binds.Bind] = pydantic.Field(
        default=frozenset()
    )
    '''A set of mouse input features to poll for.'''

    controller_whitelist: frozenset[controller_binds.Bind] = pydantic.Field(
        default=frozenset()
    )
    '''A set of controller input features to poll for.'''

    capture_binds: frozenset[base_bind.Bind] = pydantic.Field(
        default=frozenset([
            unified_binds.Binds.MOUSE_LEFT, 
            unified_binds.Binds.MOUSE_RIGHT
        ])
    )
    '''A set of binds that enable data capturing when activated.'''

    capture_bind_gate: validated_types.GateCallable = pydantic.Field(default=any)
    '''Whether any or all of the capture binds must be active to enable capturing.'''

    ignore_empty_polls: bool = pydantic.Field(default=True)
    '''Whether or not empty rows of features should be written to the data.'''

    reset_mouse_on_release: bool = pydantic.Field(default=True)
    '''Whether or not mouse deltas are reset to 0 when the capture bind is released.'''

    @property
    def whitelist(self) -> list[base_bind.Bind]:
        '''All device whitelists combined.'''
        return self.keyboard_whitelist + self.mouse_whitelist + self.controller_whitelist
    
    @property
    def features_per_poll(self) -> int:
        '''The number of features in each poll (row).'''
        return len(self.keyboard_whitelist + self.mouse_whitelist + self.controller_whitelist)
