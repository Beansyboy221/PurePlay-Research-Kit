import pydantic

from data.polling import bind_enums
from data.processing import data_params

class CollectConfig(data_params.ResolvedDataParams):
    '''Fields expected for collect mode to work properly.'''
    # Should this be inherited/composed and draw the name from the mode?
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        for field_name, field_info in cls.model_fields.items():
            if field_info.validation_alias is None:
                field_info.validation_alias = pydantic.AliasPath('collect', field_name) # How can I implement subsections in the config?
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

    save_dir: pydantic.DirectoryPath = pydantic.Field(default='data')
    '''The directory to save the data to.'''


