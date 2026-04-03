import argparse
import pydantic

from prompters import (
    prompter_utils,
    base_prompter
)

from . import (
    startup_config,
    global_logger
)

def parse_args() -> argparse.Namespace:
    '''Parses command-line arguments.'''
    parser = argparse.ArgumentParser(description='PurePlay-Research-Kit')
    # Can these be easily auto generated from the commandline_args?
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='The path to your app config file.'
    )
    parser.add_argument(
        '-g', '-use_gui',
        type=bool,
        help='Add this flag to use GUI instead of CLI',
        default=False
    )
    parser.add_argument(
        '-l', '-log_level',
        type=str,
        help=f'The logging level of the program.',
        choices=[log_level for log_level in global_logger.LogLevel],
        default=global_logger.LogLevel.INFO
    )
    return parser.parse_args()

def load_config_file(config_path: str) -> dict:
    '''Loads a toml config file and returns a dictionary.'''
    import tomllib
    with open(config_path, 'rb') as file_handle:
        return tomllib.load(file_handle)

def get_global_configs(args: argparse.Namespace = None) -> tuple[str, bool]:
    '''Resolves global app configs from cli or gui input.'''
    config = startup_config.StartupConfig.model_validate(vars(args))
    if config.config_path:
        return config
    
    prompter: base_prompter.BasePrompter = prompter_utils.AVAILABLE_PROMPTERS.get(pydantic.FilePath)
    if not prompter:
        raise RuntimeError('No prompter registered for type: pydantic.FilePath')
    
    config_prompt_args = {
        'title': 'Config File',
        'description': 'Input your config file path...',
    }
    if config.use_gui: # Would it be better to just pass a prompter strategy?
        config.config_path = prompter.prompt_with_gui(**config_prompt_args)
    else:
        config.config_path = prompter.prompt_with_cli(**config_prompt_args)
        config.use_gui = False
    return config

def fill_missing_fields(
        config_class: type[pydantic.BaseModel], 
        input_configs: dict = {}, 
        use_gui: bool = True
    ) -> dict:
    '''Generates guis or cli prompts to populate missing config fields.'''
    for field_name, field_info in config_class.model_fields.items():
        if field_name in input_configs or not field_info.is_required():
            continue
        
        prompter: base_prompter.BasePrompter = prompter_utils.AVAILABLE_PROMPTERS.get(field_info.annotation)
        if not prompter:
            raise RuntimeError(f'No prompter registered for type: {field_info.annotation}')
        
        prompt_method = prompter.prompt_with_gui if use_gui else prompter.prompt_with_cli
        input_configs[field_name] = prompt_method(
            title=field_name,
            description=field_info.description
        )
    return input_configs