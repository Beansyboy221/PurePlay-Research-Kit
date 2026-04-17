import typing

from globals import formats
from . import cli_prompter

#region Common
@cli_prompter.register(str, 'Enter a string.')
@cli_prompter.register(int, 'Enter an integer.')
@cli_prompter.register(float, 'Enter a float.')
@cli_prompter.register(complex, 'Enter a complex number.')
def prompt_for_str_castable(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a string castable.'
    ):
    return data_type(input(f'\n{title}\n{message}\n'))

@cli_prompter.register(bool, 'Enter a boolean.')
def prompt_for_bool(data_type: type, title: str, message: str):
    return formats.BOOL_STRINGS[
        input(f'\n{title}\n{message}\n{formats.BOOL_STRINGS}\n')
    ]
#endregion

#region Collections
@cli_prompter.register(set, 'Please enter a set of values.')
@cli_prompter.register(list, 'Please enter a list of values.')
@cli_prompter.register(tuple, 'Please enter a tuple of values.')
@cli_prompter.register(frozenset, 'Please enter a frozenset of values.')
def prompt_for_sequence(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a sequence of values.'
    ):
    print(f'\n{title}\n{message}\n')
    args = typing.get_args(data_type)
    list_item_type = args[0] if args else str
    output = []
    while True:
        if not list_item_type:
            temp_item_type: str = cli_prompter.prompt_with_cli(
                data_type=str, 
                title='Data Type', 
                message=f'Please input the type of element {len(output)}.'
            )
        input = cli_prompter.prompt_with_cli(list_item_type or temp_item_type)
        if input is None:
            break
        output.append(input)
    return data_type[output]

@cli_prompter.register(dict, 'Please enter a dictionary of values.')
def prompt_for_dict(data_type: type, title: str, message: str):
    print(f'\n{title}\n{message}\n')
    args = typing.get_args(data_type)
    key_type = args[0] if args else str
    value_type = args[1] if len(args) > 1 else str
    output = {}
    while True:
        key = cli_prompter.prompt_with_cli(key_type, message='Please enter a key for a new pair.')
        if key is None:
            break
        value = cli_prompter.prompt_with_cli(value_type, message='Please enter a value for the pair.')
        if value is None:
            break
        output[key] = value
    return output
#endregion