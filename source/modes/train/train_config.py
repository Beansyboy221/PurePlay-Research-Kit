import pydantic

from models import model_params
from source.data.polling import (
    poll_params
)
from source.data.polling import bind_enums
from . import (
    tuning_overrides,
    search_spaces
)

class TrainConfig(
        poll_params.PollParams,
        model_params.ModelParams,
        search_spaces.SearchSpaces,
        tuning_overrides.TuningOverrides
    ):
    '''Fields expected for train mode to work properly.'''
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for field_name, field_info in cls.model_fields.items():
            if field_info.validation_alias is None:
                field_info.validation_alias = pydantic.AliasPath('collect', field_name)
        cls.model_rebuild(force=True)
    
    kill_bind_list: frozenset[bind_enums.DigitalBind] = pydantic.Field(
        default=frozenset([bind_enums.DigitalBind.ESC]),
        min_length=1
	)
    '''A set of binds that stop the program.'''
    
    kill_bind_logic: bind_enums.BindGate = pydantic.Field(
        default=bind_enums.BindGate.ANY
	)
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