import pydantic

from misc import validated_types
import modes
from models import params as model_params
from polling import (
    unified_binds,
    base_bind,
    params as poll_params
)
from . import (
    tuning_overrides,
    search_spaces
)

class TrainConfig(
        modes.ModeConfig,
        poll_params.PollParams,
        model_params.ModelParams,
        search_spaces.SearchSpaces,
        tuning_overrides.TuningOverrides
    ):
    '''Fields expected for train mode to work properly.'''
    kill_bind_list: frozenset[base_bind.Bind] = pydantic.Field(
        default=frozenset([unified_binds.Binds.ESC]),
        min_length=1
	)
    '''A set of binds that stop the program when activated.'''
    
    kill_bind_logic: validated_types.GateCallable = pydantic.Field(default=any)
    '''Whether any or all of the kill binds must be held to stop.'''

    save_dir: pydantic.DirectoryPath = pydantic.Field(default='models')
    '''The directory to save the models to.'''

    training_file_dir: pydantic.DirectoryPath = pydantic.Field(
        default='data\\train\\benign'
	)
    '''The directory containing the training files.'''

    validation_file_dir: pydantic.DirectoryPath = pydantic.Field(
        default='data\\val\\benign'
	)
    '''The directory containing the validation files.'''

    cheat_training_file_dir: pydantic.DirectoryPath | None = pydantic.Field(
        default='data\\train\\cheat'
	)
    '''The directory containing the cheat training files.'''

    cheat_validation_file_dir: pydantic.DirectoryPath | None = pydantic.Field(
        default='data\\val\\cheat'
	)
    '''The directory containing the cheat validation files.'''

    batch_size_tune_epochs: int = pydantic.Field(default=50, ge=1)
    '''The number of epochs to tune the batch size for.'''