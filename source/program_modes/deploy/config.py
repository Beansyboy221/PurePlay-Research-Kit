import pydantic

from globals.enums import binds
from . import enums

class ModeConfig(pydantic.BaseModel):
    '''Fields expected for deploy mode to work properly.'''
    
    kill_bind_list: frozenset[binds.DigitalBind] = pydantic.Field(
        default_factory=lambda: [binds.DigitalBind.ESC],
        min_items=1,
        description='A set of binds that stop the program.',
    )
    '''A set of binds that stop the program.'''
    
    kill_bind_logic: binds.BindGate = pydantic.Field(
        default=binds.BindGate.ANY,
        description='Whether any or all of the kill binds must be held to stop.',
    )
    '''Whether any or all of the kill binds must be held to stop.'''

    model_file: pydantic.FilePath = pydantic.Field(
        description='The path to the model file.',
        validation_alias=pydantic.AliasPath('deploy', 'model_file'),
        json_schema_extra={'file_types': [('Checkpoint Files', '*.ckpt'),]}
    )
    '''The path to the model file.'''

    deployment_window_type: enums.WindowType = pydantic.Field(
        default=enums.WindowType.TUMBLING,
        description='The type of window to use for deployment.',
        validation_alias=pydantic.AliasPath('deploy', 'deployment_window_type')
    )
    '''The type of window to use for deployment.'''

    write_to_file: bool = pydantic.Field(
        default=True,
        description='Whether or not to write the data to a file.',
        validation_alias=pydantic.AliasPath('deploy', 'write_to_file')
    )
    '''Whether or not to write the data to a file.'''

    save_dir: pydantic.DirectoryPath = pydantic.Field(
        default='data',
        description='The directory to save the data to.',
        validation_alias=pydantic.AliasPath('deploy', 'save_dir')
    )
    '''The directory to save the data to.'''

    capture_bind_list: frozenset[binds.DigitalBind] = pydantic.Field(
        default_factory=lambda: [binds.DigitalBind.LEFT_MOUSE, binds.DigitalBind.RIGHT_MOUSE],
        description='A set of binds that enable data capturing when held.',
        validation_alias=pydantic.AliasPath('deploy', 'capture_bind_list')
    )
    '''A set of binds that enable data capturing when held.'''

    capture_bind_logic: binds.BindGate = pydantic.Field(
        default=binds.BindGate.ANY, 
        description='Whether any or all of the capture binds must be held to enable capturing.',
        validation_alias=pydantic.AliasPath('deploy', 'capture_bind_logic')
    )
    '''Whether any or all of the capture binds must be held to enable capturing.'''