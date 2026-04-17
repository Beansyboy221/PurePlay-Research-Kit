import typing

from ....utilities.prompt_utils.cli import cli_prompter

@cli_prompter.register(typing.Union, 'N/A')
@cli_prompter.register(typing.Annotated, 'N/A')
def prompt_for_subscripted(data_type: type, title: str, message: str):
    return cli_prompter.prompt_with_cli(typing.get_origin(data_type))