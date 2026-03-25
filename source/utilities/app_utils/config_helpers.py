import argparse
import pydantic

from prompters import (
    prompter_utils,
    base_prompter
)

from . import global_logger

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
    '''Loads a toml config file and returns a dictionary.'''
    import tomllib
    with open(config_path, 'rb') as file_handle:
        return tomllib.load(file_handle)

def get_global_configs(args: argparse.Namespace = None) -> tuple[str, bool]:
    '''Resolves global app configs from cli or gui input.'''
    config_path = getattr(args, 'config', None)
    use_gui = getattr(args, 'gui', None)
    log_level = getattr(args, 'log_level', global_logger.LogLevel.INFO)
    if config_path:
        return config_path, use_gui, log_level
    
    prompter: base_prompter.BasePrompter = prompter_utils.AVAILABLE_PROMPTERS.get(pydantic.FilePath)
    if not prompter:
        raise RuntimeError(f'No prompter registered for type: {pydantic.FilePath}')
    
    config_prompt_args = {
        'title': 'Config File',
        'description': 'Input your config file path...',
    }
    if use_gui: # Would it be better to just pass a prompter strategy?
        config_path = prompter.prompt_with_gui(**config_prompt_args)
    else:
        config_path = prompter.prompt_with_cli(**config_prompt_args)
        use_gui = False
    return config_path, use_gui, log_level

def populate_missing_fields(
        config_class: type[pydantic.BaseModel], 
        data_dict: dict = {}, 
        use_gui: bool = True
    ) -> dict:
    '''Generates guis or cli prompts to populate missing config fields.'''
    for field_name, field_info in config_class.model_fields.items():
        if field_name in data_dict or not field_info.is_required():
            continue
        
        extras = field_info.json_schema_extra
        override = extras.get('data_type_override')
        data_type = override if override else field_info.annotation

        prompter = prompter_utils.AVAILABLE_PROMPTERS.get(data_type)
        if not prompter:
            raise RuntimeError(f'No prompter registered for type: {data_type}')
        prompt_method = prompter.prompt_with_gui if use_gui else prompter.prompt_with_cli
    
        data_dict[field_name] = prompt_method(
            title=field_name,
            description=field_info.description,
            data_type=data_type,
            options=extras.get('options'),
            is_file=data_type is pydantic.FilePath,
            is_dir=data_type is pydantic.DirectoryPath,
            file_types=extras.get('file_types')
        )
    return data_dict