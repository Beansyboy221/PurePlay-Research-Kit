import argparse
import pydantic

# PurePlay imports
from source.globals.constants import tunables
from source.utilities.model_utils import (
    modelregistry,
    basemodel
)
from source.globals import global_logger
from source.utilities.app_utils import (
    input_utils
)

def parse_args() -> argparse.Namespace:
    '''Parses command-line arguments.'''
    parser = argparse.ArgumentParser(description='PurePlay-Research-Kit')
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to config file (.toml, .json, .yaml, etc.)',
    )
    parser.add_argument(
        '-g', '-gui',
        type=bool,
        help='Add this flag to use GUI instead of CLI'
    )
    parser.add_argument(
        '-l', '-log_level',
        type=str,
        help='The logging level of the program (DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL)'
    )
    return parser.parse_args()

def load_config_file(config_path: str) -> dict:
    '''Loads a config file and returns a dictionary.'''
    from source.utilities.file_utils.reading import reader_registry
    reader = reader_registry.get_reader(config_path)
    read_method = reader.get_read_method(dict)
    if not read_method:
        raise ValueError(f'Reader for: {config_path} does not support reading dicts.')
    return read_method(config_path)

def get_global_configs(args: object = None) -> tuple[str, bool]:
    '''Resolves global app configs from cli or gui input.'''
    config_path = args.config if args else None
    use_gui = args.gui if args else None
    log_level = args.log_level if args else global_logger.LogLevel.INFO
    if config_path:
        return config_path, use_gui
    config_prompt_args = {
        'title': 'Config File',
        'description': 'Input your config file path...',
        'is_file': True,
        'file_types': [('Config Files', '*.json *.toml *.yaml *.yml *.ini *.conf')]
    }
    if use_gui is None:
        config_path = input_utils.get_input_from_cli(**config_prompt_args)
        use_gui = False
    else:
        config_path = input_utils.get_input_from_gui(**config_prompt_args)
        use_gui = True
    return config_path, use_gui, log_level

def populate_missing_fields(config_class: type[pydantic.BaseModel], data_dict: dict = {}, use_gui: bool = True) -> dict:
    '''Generates guis or cli prompts to populate missing config fields.'''
    if use_gui:
        input_method = input_utils.get_input_from_gui
    else:
        input_method = input_utils.get_input_from_cli
        
    for field_name, field_info in config_class.model_fields.items():
        if field_name in data_dict or not field_info.is_required():
            continue
        
        extras = field_info.json_schema_extra
        override = extras.get('data_type_override')
        data_type = override if override else field_info.annotation

        data_dict[field_name] = input_method(
            title=field_name,
            description=field_info.description,
            data_type=data_type,
            options=extras.get('options'),
            is_file=data_type is pydantic.FilePath,
            is_dir=data_type is pydantic.DirectoryPath,
            file_types=extras.get('file_types')
        )
    return data_dict

#region Validators
def validate_model_name(model_name: str) -> type[basemodel.BaseModel]:
    if model_name not in modelregistry.AVAILABLE_MODELS:
        raise ValueError(f"Invalid model class name '{model_name}'. Options: {list(basemodel.AVAILABLE_MODELS.keys())}")
    return modelregistry.AVAILABLE_MODELS[model_name]

def validate_scaler_name(scaler_name: str) -> type[object]:
    if scaler_name not in tunables.SCALER_MAP:
        raise ValueError(f"Invalid scaler name '{scaler_name}'. Options: {list(tunables.SCALER_MAP.keys())}")
    return tunables.SCALER_MAP[scaler_name]

def validate_optimizer_name(optimizer_name: str) -> type[object]:
    if optimizer_name not in tunables.OPTIMIZER_MAP:
        raise ValueError(f"Invalid optimizer name '{optimizer_name}'. Options: {list(tunables.OPTIMIZER_MAP.keys())}")
    return tunables.OPTIMIZER_MAP[optimizer_name]

def validate_scheduler_name(scheduler_name: str) -> type[object]:
    if scheduler_name not in tunables.SCHEDULER_MAP:
        raise ValueError(f"Invalid scheduler name '{scheduler_name}'. Options: {list(tunables.SCHEDULER_MAP.keys())}")
    return tunables.SCHEDULER_MAP[scheduler_name]
#endregion

