import sklearn.base

from .....prompt_utils.cli import cli_prompter

@cli_prompter.register(sklearn.base.TransformerMixin)
def prompt_for_transformer(
        data_type: type, 
        title: str, 
        message: str = 'Please select a transformer.'
    ):
    raise NotImplementedError