import pydantic

import prompting
from prompting.cli import helpers

def load_config_file(config_path: str) -> dict:
    '''Loads a toml config file and returns a dictionary.'''
    import tomllib
    with open(config_path, 'rb') as file_handle:
        return tomllib.load(file_handle)

def fill_missing_fields(
        config_class: type[pydantic.BaseModel], 
        input_configs: dict = {}, 
        prompt_method: prompting.PromptMethod = helpers.create_prompt,
    ) -> dict:
    '''Generates guis or cli prompts to populate missing config fields.'''
    for field_name, field_info in config_class.model_fields.items():
        if field_name in input_configs or not field_info.is_required():
            continue
        input_configs[field_name] = prompt_method(
            title=field_name,
            description=field_info.description
        )
    return input_configs