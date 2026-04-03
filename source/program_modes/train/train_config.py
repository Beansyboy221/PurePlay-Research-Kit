import pydantic

from ml_models import model_params
from utilities.poll_utils import (
    poll_params,
    bind_enums
)
from . import (
    search_spaces,
    tuning_overrides
)

class TrainConfig(
        pydantic.BaseModel,
        poll_params.PollParams,
        model_params.ModelParams,
        search_spaces.SearchSpaces,
        tuning_overrides.TuningOverrides
    ):
    '''Fields expected for train mode to work properly.'''
    kill_bind_list: frozenset[bind_enums.DigitalBind] = pydantic.Field(
        default_factory=lambda: [bind_enums.DigitalBind.ESC],
        min_items=1,
        description='A set of binds that stop the program.',
    )
    '''A set of binds that stop the program.'''
    
    kill_bind_logic: bind_enums.BindGate = pydantic.Field(
        default=bind_enums.BindGate.ANY,
        description='Whether any or all of the kill binds must be held to stop.',
    )
    '''Whether any or all of the kill binds must be held to stop.'''

    save_dir: pydantic.DirectoryPath = pydantic.Field(
        default='models',
        description='The directory to save the models to.',
        validation_alias=pydantic.AliasPath('train', 'save_dir')
    )
    '''The directory to save the models to.'''

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

    batch_size_tune_epochs: int = pydantic.Field(
        default=50,
        ge=1,
        description='The number of epochs to tune the batch size for.',
        validation_alias=pydantic.AliasPath('train', 'batch_size_tune_epochs')
    )
    '''The number of epochs to tune the batch size for.'''