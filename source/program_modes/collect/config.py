import pydantic

from utilities.data_utils import data_params
from utilities.poll_utils import (
    poll_params,
    bind_enums
)

class ModeConfig(
        pydantic.BaseModel,
        data_params.ResolvedDataParams,
        poll_params.PollParams
    ):
    '''Fields expected for collect mode to work properly.'''
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

    # Capture fields should be separate along with try_poll
    capture_binds: frozenset[bind_enums.DigitalBind] = pydantic.Field(
        default_factory=lambda: [bind_enums.DigitalBind.LEFT_MOUSE, bind_enums.DigitalBind.RIGHT_MOUSE],
        description='A set of binds that enable data capturing when held.',
        validation_alias=pydantic.AliasPath('deploy', 'capture_binds')
    )
    '''A set of binds that enable data capturing when held.'''

    capture_bind_gate: bind_enums.BindGate = pydantic.Field(
        default=bind_enums.BindGate.ANY, 
        description='Whether any or all of the capture binds must be held to enable capturing.',
        validation_alias=pydantic.AliasPath('deploy', 'capture_bind_gate')
    )
    '''Whether any or all of the capture binds must be held to enable capturing.'''

    save_dir: pydantic.DirectoryPath = pydantic.Field(
        default='data',
        description='The directory to save the data to.',
        validation_alias=pydantic.AliasPath('collect', 'save_dir')
    )
    '''The directory to save the data to.'''
