import pydantic

from globals.enums import binds
from source.ml_models import (
    model_utils,
    base_model
)

class ModeConfig(pydantic.BaseModel):
    '''Fields expected for train mode to work properly.'''
    
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

    save_dir: pydantic.DirectoryPath = pydantic.Field(
        default='models',
        description='The directory to save the models to.',
        validation_alias=pydantic.AliasPath('train', 'save_dir')
    )
    '''The directory to save the models to.'''

    model_class: base_model.BaseModel = pydantic.Field(
        description='The model class name to use for training.',
        validation_alias=pydantic.AliasPath('train', 'model_class'),
        json_schema_extra={
            'options': model_utils.AVAILABLE_MODELS
        }
    )
    '''The model class name to use for training.'''

    ignore_empty_polls: bool = pydantic.Field(
        default=True, 
        description='Whether or not empty rows of features should be ignored by the model.',
        validation_alias=pydantic.AliasPath('train', 'ignore_empty_polls')
    )
    '''Whether or not empty rows of features should be ignored by the model.'''

    polls_per_window: int = pydantic.Field(
        default=128,
        multiple_of=2,
        ge=8, 
        description='The number of polls(rows) of the whitelisted features in each window.',
        validation_alias=pydantic.AliasPath('train', 'polls_per_window')
    )
    '''The number of polls(rows) of the whitelisted features in each window.'''

    window_stride: int = pydantic.Field(
        default=polls_per_window // 2,
        ge=1,
        lt=polls_per_window,
        description='The number of polls(rows) to skip between windows.',
        validation_alias=pydantic.AliasPath('train', 'window_stride')
    )
    '''The number of polls(rows) to skip between windows.'''

    windows_per_batch: int | None = pydantic.Field(
        default=None,
        multiple_of=2,
        ge=16,
        description='The number of windows in each batch.',
        validation_alias=pydantic.AliasPath('train', 'windows_per_batch')
    )
    '''The number of windows in each batch.'''

    training_file_dir: pydantic.DirectoryPath = pydantic.Field(
        default='data\\train\\benign',
        description='The directory containing the training files.',
        validation_alias=pydantic.AliasPath('train', 'training_file_dir')
    )
    '''The directory containing the training files.'''

    validation_file_dir: pydantic.DirectoryPath = pydantic.Field(
        default='data\\val\\benign',
        description='The directory containing the validation files.',
        validation_alias=pydantic.AliasPath('train', 'validation_file_dir')
    )
    '''The directory containing the validation files.'''

    cheat_training_file_dir: pydantic.DirectoryPath | None = pydantic.Field(
        default='data\\train\\cheat',
        description='The directory containing the cheat training files.',
        validation_alias=pydantic.AliasPath('train', 'cheat_training_file_dir')
    )
    '''The directory containing the cheat training files.'''

    cheat_validation_file_dir: pydantic.DirectoryPath | None = pydantic.Field(
        default='data\\val\\cheat',
        description='The directory containing the cheat validation files.',
        validation_alias=pydantic.AliasPath('train', 'cheat_validation_file_dir')
    )
    '''The directory containing the cheat validation files.'''

    keyboard_whitelist: frozenset[binds.KeyBind] = pydantic.Field(
        default_factory=lambda: [],
        description='A set of all keyboard input features in the data.',
        validation_alias=pydantic.AliasPath('train', 'keyboard_whitelist')
    )
    '''A set of all keyboard input features in the data.'''

    mouse_whitelist: frozenset[binds.MouseBind] = pydantic.Field(
        default_factory=lambda: [],
        description='A set of all mouse input features in the data.',
        validation_alias=pydantic.AliasPath('train', 'mouse_whitelist')
    )
    '''A set of all mouse input features in the data.'''

    gamepad_whitelist: frozenset[binds.GamepadBind] = pydantic.Field(
        default_factory=lambda: [],
        description='A set of all gamepad input features in the data.',
        validation_alias=pydantic.AliasPath('train', 'gamepad_whitelist')
    )
    '''A set of all gamepad input features in the data.'''