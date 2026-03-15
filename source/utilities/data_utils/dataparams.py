import pydantic

# PurePlay imports
from globals.enums import binds

class DataParams(pydantic.BaseModel):
    '''Parameters that must be defined by the user.'''
    keyboard_whitelist: list[binds.KeyBind] = pydantic.Field(default_factory=list)
    mouse_whitelist: list[binds.MouseBind] = pydantic.Field(default_factory=list)
    gamepad_whitelist: list[binds.GamepadBind] = pydantic.Field(default_factory=list)
    ignore_empty_polls: bool = True
    polls_per_sequence: int = pydantic.Field(gt=0)

    @property
    def whitelist(self) -> list[binds.Bind]:
        return self.keyboard_whitelist + self.mouse_whitelist + self.gamepad_whitelist

    @property
    def features_per_poll(self) -> int:
        return len(self.keyboard_whitelist + self.mouse_whitelist + self.gamepad_whitelist)

class ResolvedDataParams(DataParams):
    '''DataParams extended with properties sourced from the dataset itself.'''
    polling_rate: int = pydantic.Field(gt=0)
    reset_mouse_on_release: bool = True