import typing

from prompters import (
    prompter_utils,
    base_prompter
)
from utilities.app_utils import startup_utils

from . import (
    mode_utils,
    base_mode
)

def select_and_run_mode(config_dict: dict, use_gui: bool = False) -> None:
    '''Runs a mode selected by user prompting.'''
    prompter: base_prompter.BasePrompter = prompter_utils.AVAILABLE_PROMPTERS.get(mode_utils.ProgramMode)
    if not prompter:
        raise RuntimeError(f'No prompter registered for type: {base_mode.ProgramMode}')

    mode_selection_kwargs = {
        'title': 'Program Mode',
        'description': 'Please select the mode you want to run...'
    }
    if use_gui:
        mode_name = prompter.prompt_with_gui(**mode_selection_kwargs)
    else:
        mode_name = prompter.prompt_with_cli(**mode_selection_kwargs)
    mode: type[base_mode.ProgramMode] = mode_utils.AVAILABLE_MODES.get(mode_name)
    if not mode:
        raise RuntimeError(f'No mode registered for name: {mode_name}')
    
    populated_configs = []
    config_dict = startup_utils.fill_missing_fields(mode.config, config_dict, use_gui)
    populated_configs.append(mode.config.model_validate(config_dict))
    mode.entry_point(**populated_configs)