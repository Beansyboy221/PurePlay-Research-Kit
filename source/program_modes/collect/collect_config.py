import pydantic

from utilities.data_utils import data_params
from utilities.poll_utils import (
    poll_params,
    bind_enums
	)

class CollectConfig(
        data_params.ResolvedDataParams,
        poll_params.PollParams
    ):
    '''Fields expected for collect mode to work properly.'''
    # Should this be inherited/composed and draw the name from the mode?
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for field_name, field_info in cls.model_fields.items():
            if field_info.validation_alias is None:
                field_info.validation_alias = pydantic.AliasPath('collect', field_name) # How can I implement subsections in the config?
        cls.model_rebuild(force=True)
    
    kill_bind_list: frozenset[bind_enums.DigitalBind] = pydantic.Field(
        default_factory=lambda: [bind_enums.DigitalBind.ESC],
        min_items=1
    )
    '''A set of binds that stop the program.'''
    
    kill_bind_logic: bind_enums.BindGate = pydantic.Field(
        default=bind_enums.BindGate.ANY
    )
    '''Whether any or all of the kill binds must be held to stop.'''

    save_dir: pydantic.DirectoryPath = pydantic.Field(default='data')
    '''The directory to save the data to.'''


