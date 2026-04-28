from models import (
    helpers,
    base
)
from prompting import cli

@cli.register(base.BaseModel, 'Please select a model.')
def prompt_for_base_model(data_type: type, title: str, message: str):
    return input(f'\n{title}\n{message}\nOptions: {helpers.AVAILABLE_MODELS.keys()}')