from startup import startup_helpers
from prompting.cli import cli_prompter
from prompting.tkinter import tkinter_prompter
from . import base_mode

def select_and_run_mode(config_dict: dict, use_gui: bool = False) -> None:
    '''Runs a mode selected by user prompting.'''
    mode_selection_kwargs = {
        'data_type': base_mode.ProgramMode,
        'title': 'Program Mode',
        'message': 'Please select the mode you want to run...'
    }
    if use_gui:
        prompt_method = tkinter_prompter.prompt_with_tkinter
    else:
        prompt_method = cli_prompter.prompt_with_cli
    mode: base_mode.ProgramMode = prompt_method(**mode_selection_kwargs)
    config_dict = startup_helpers.fill_missing_fields(
        config_class=mode.config, 
        input_configs=config_dict, 
        prompt_method=prompt_method
    )
    mode.entry_point(mode.config.model_validate(config_dict))