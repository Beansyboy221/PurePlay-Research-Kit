import pydantic

from utilities.poll_utils import (
    poll_params,
    bind_enums
)

class DeployConfig(poll_params.PollParams):
    '''Fields expected for deploy mode to work properly.'''
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for field_name, field_info in cls.model_fields.items():
            if field_info.validation_alias is None:
                field_info.validation_alias = pydantic.AliasPath('deploy', field_name)
        cls.model_rebuild(force=True)
    
    model_file: pydantic.FilePath = # Create new checkpoint file path type or use a validator
    '''The path to the model file.'''

    write_to_file: bool = pydantic.Field(default=True)
    '''Whether or not to write the data to a file.'''

    save_dir: pydantic.DirectoryPath = pydantic.Field(default='data')
    '''The directory to save the data to.'''

    kill_bind_list: frozenset[bind_enums.DigitalBind] = pydantic.Field(
        default_factory=lambda: [bind_enums.DigitalBind.ESC],
        min_items=1,
    )
    '''A set of binds that stop the program.'''
    
    kill_bind_logic: bind_enums.BindGate = pydantic.Field(
        default=bind_enums.BindGate.ANY
    )
    '''Whether any or all of the kill binds must be held to stop.'''