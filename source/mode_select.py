import importlib
import pydantic
import pathlib
import typing

# PurePlay imports
from source.globals import global_logger
from source.utilities.app_utils import (
    config_utils,
    input_utils
)

def select_and_run_mode(config_dict: dict, use_gui: bool = False) -> None:
    '''Main entry point. Runs a mode given by user input.'''
    registry = _discover_modes()
    mode_selection_kwargs = {
        'title': 'Mode',
        'description': 'Please select the mode you want to run...',
        'data_type': type[function],
        'options': list(registry.keys())
    }
    if use_gui:
        mode_name = input_utils.get_input_from_gui(**mode_selection_kwargs)
    else:
        mode_name = input_utils.get_input_from_cli(**mode_selection_kwargs)
    entry_point, config_classes = registry[mode_name]
    _run_mode(entry_point, config_classes, config_dict, use_gui)

def _discover_modes() -> dict[str, tuple]:
    '''Scans the modes package and loads all mode.py manifests.'''
    registry = {}
    modes_dir = pathlib.Path(__file__).parent / 'modes'
    for mode_file in modes_dir.rglob('mode.py'):
        mode_name = mode_file.parent.name
        try:
            manifest = importlib.import_module(f'modes.{mode_name}.mode')
            registry[mode_name] = (manifest.ENTRY_POINT, manifest.CONFIG_CLASSES)
        except AttributeError as e:
            raise RuntimeError(f'Invalid mode manifest in modes.{mode_name}.mode: {e}')
    return registry

def _run_mode(
        entry_point: callable, 
        config_classes: tuple[pydantic.BaseModel] | pydantic.BaseModel, 
        config_dict: dict[str, typing.Any], 
        use_gui: bool = False
    ) -> None:
    '''Dynamically loads and executes the selected mode.'''
    populated_configs = []
    for config_class in config_classes:
        config_dict = config_utils.populate_missing_fields(
            config_class, 
            config_dict, 
            use_gui=use_gui
        )
        populated_configs.append(config_class.model_validate(config_dict))
    entry_point(**populated_configs)

if __name__ == '__main__':
    try:
        args = config_utils.parse_args()
        config_path, use_gui, log_level = config_utils.get_global_configs(args)
        global_logger.set_log_level(log_level)
        config_dict = config_utils.load_config_file(config_path)
        select_and_run_mode(config_dict, use_gui=use_gui)
    except Exception as e:
        global_logger.exception(e)