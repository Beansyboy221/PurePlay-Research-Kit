import typing

from prompting.cli import helpers

@helpers.register(typing.Union)
@helpers.register(typing.Generic)
@helpers.register(typing.Callable)
@helpers.register(typing.Annotated)
def prompt_for_generic(data_type: type, title: str, message: str):
    return helpers.create_prompt(typing.get_origin(data_type))