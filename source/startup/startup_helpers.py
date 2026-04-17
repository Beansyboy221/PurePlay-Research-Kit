import pydantic
import typing

from prompting.cli import cli_prompter

def load_config_file(config_path: str) -> dict:
    '''Loads a toml config file and returns a dictionary.'''
    import tomllib
    with open(config_path, 'rb') as file_handle:
        return tomllib.load(file_handle)

def fill_missing_fields(
        config_class: type[pydantic.BaseModel], 
        input_configs: dict = {}, 
        prompt_method: typing.Callable[[type, str, str], typing.Any] = cli_prompter.prompt_with_cli, # Should this type hint be its own specific type called PromptMethod? 
    ) -> dict:                                                                                       # What would this mean for where prompting methods live?
    '''Generates guis or cli prompts to populate missing config fields.'''
    for field_name, field_info in config_class.model_fields.items():
        if field_name in input_configs or not field_info.is_required():
            continue
        input_configs[field_name] = prompt_method(
            title=field_name,
            description=field_info.description
        )
    return input_configs