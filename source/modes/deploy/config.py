import pydantic

from misc import validated_types
import modes
from polling import (
    unified_binds,
    base_bind,
    params
)

class DeployConfig(modes.ModeConfig, params.PollParams):
    '''Fields expected for deploy mode to work properly.'''
    model_file: validated_types.CheckpointPath
    '''The path to the model file.'''

    write_to_file: bool = pydantic.Field(default=True)
    '''Whether or not to write the data to a file.'''

    save_dir: pydantic.DirectoryPath = pydantic.Field(default='data')
    '''The directory to save the data to.'''

    kill_bind_list: frozenset[base_bind.Bind] = pydantic.Field(
        default=frozenset([unified_binds.Binds.ESC]),
        min_length=1
    )
    '''A set of binds that stop the program when activated.'''
    
    kill_bind_logic: validated_types.GateCallable = pydantic.Field(default=any)
    '''Whether any or all of the kill binds must be held to stop.'''