import typing

from prompters import (
    prompter_utils,
    base_prompter
)
from utilities.app_utils import (
    config_helpers,
    global_logger,
    mode_utils
)

def select_and_run_mode(config_dict: dict, use_gui: bool = False) -> None:
    '''Main entry point. Runs a mode given by user input.'''
    prompter: base_prompter.BasePrompter = prompter_utils.AVAILABLE_PROMPTERS.get(mode_utils.ProgramMode)
    if not prompter:
        raise RuntimeError(f'No prompter registered for type: {mode_utils.ProgramMode}')

    mode_selection_kwargs = {
        'title': 'Program Mode',
        'description': 'Please select the mode you want to run...'
    }
    if use_gui:
        mode_name = prompter.prompt_with_gui(**mode_selection_kwargs)
    else:
        mode_name = prompter.prompt_with_cli(**mode_selection_kwargs)
    mode = mode_utils.AVAILABLE_MODES.get(mode_name)
    if not mode:
        raise RuntimeError(f'No mode registered for name: {mode_name}')
    _run_mode(mode, config_dict, use_gui)

def _run_mode(
        mode: mode_utils.ProgramMode,
        config_dict: dict[str, typing.Any], 
        use_gui: bool = False
    ) -> None:
    '''Dynamically loads and executes the selected mode.'''
    populated_configs = []
    for config_class in mode.configs:
        config_dict = config_helpers.populate_missing_fields(
            config_class, 
            config_dict, 
            use_gui=use_gui
        )
        populated_configs.append(config_class.model_validate(config_dict))
    mode.entry_point(**populated_configs)

if __name__ == '__main__':
    try:
        args = config_helpers.parse_args()
        config_path, use_gui, log_level = config_helpers.get_global_configs(args)
        global_logger.set_log_level(log_level)
        config_dict = config_helpers.load_config_file(config_path)
        select_and_run_mode(config_dict, use_gui=use_gui)
    except Exception as e:
        global_logger.exception(e)