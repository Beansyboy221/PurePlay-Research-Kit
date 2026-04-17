from models import (
    model_registry,
    base_model
)
from prompt_utils.cli import cli_prompter

@cli_prompter.register(base_model.BaseModel, 'Please select a model.')
def prompt_for_base_model(data_type: type, title: str, message: str):
    return input(f'\n{title}\n{message}\nOptions: {model_registry.AVAILABLE_MODELS.keys()}')