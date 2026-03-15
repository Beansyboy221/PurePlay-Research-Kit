import pydantic

# PurePlay imports
from globals.enums import binds

class ModeConfig(pydantic.BaseModel):
    '''Fields expected for collect mode to work properly.'''
    
    kill_bind_list: list[binds.DigitalBind] = pydantic.Field(
        default_factory=lambda: [binds.DigitalBind.ESC],
        min_items=1,
        description='A list of binds that stop the program.',
    )
    '''A list of binds that stop the program.'''
    
    kill_bind_logic: binds.BindGate = pydantic.Field(
        default=binds.BindGate.ANY,
        description='Whether any or all of the kill binds must be held to stop.',
    )
    '''Whether any or all of the kill binds must be held to stop.'''

    save_dir: pydantic.DirectoryPath = pydantic.Field(
        default='data',
        description='The directory to save the data to.',
        validation_alias=pydantic.AliasPath('collect', 'save_dir')
    )
    '''The directory to save the data to.'''

    polling_rate: int = pydantic.Field(
        default=60, 
        gt=0,
        description='The polling rate used when polling the data (not the hardware polling rate).',
        validation_alias=pydantic.AliasPath('collect', 'polling_rate')
    )
    '''The polling rate used when polling the data (not the hardware polling rate).'''

    ignore_empty_polls: bool = pydantic.Field(
        default=True, 
        description='Whether or not empty rows of features should be written to the data.',
        validation_alias=pydantic.AliasPath('collect', 'ignore_empty_polls')
    )
    '''Whether or not empty rows of features should be written to the data.'''

    reset_mouse_on_release: bool = pydantic.Field(
        default=True,
        description='Whether or not mouse deltas are reset to 0 when the capture bind is released.',
        validation_alias=pydantic.AliasPath('collect', 'reset_mouse_on_release')
    )
    '''Whether or not mouse deltas are reset to 0 when the capture bind is released.'''
    
    capture_bind_list: list[binds.DigitalBind] = pydantic.Field(
        default_factory=lambda: [binds.DigitalBind.LEFT_MOUSE, binds.DigitalBind.RIGHT_MOUSE],
        description='A list of binds that enable data capturing when held.',
        validation_alias=pydantic.AliasPath('collect', 'capture_bind_list')
    )
    '''A list of binds that enable data capturing when held.'''

    capture_bind_logic: binds.BindGate = pydantic.Field(
        default=binds.BindGate.ANY, 
        description='Whether any or all of the capture binds must be held to enable capturing.',
        validation_alias=pydantic.AliasPath('collect', 'capture_bind_logic')
    )
    '''Whether any or all of the capture binds must be held to enable capturing.'''

    keyboard_whitelist: list[binds.KeyBind] = pydantic.Field(
        default_factory=lambda: [],
        description='A list of all keyboard input features in the data.',
        validation_alias=pydantic.AliasPath('collect', 'keyboard_whitelist')
    )
    '''A list of all keyboard input features in the data.'''

    mouse_whitelist: list[binds.MouseBind] = pydantic.Field(
        default_factory=lambda: [],
        description='A list of all mouse input features in the data.',
        validation_alias=pydantic.AliasPath('collect', 'mouse_whitelist')
    )
    '''A list of all mouse input features in the data.'''

    gamepad_whitelist: list[binds.GamepadBind] = pydantic.Field(
        default_factory=lambda: [],
        description='A list of all gamepad input features in the data.',
        validation_alias=pydantic.AliasPath('collect', 'gamepad_whitelist')
    )
    '''A list of all gamepad input features in the data.'''