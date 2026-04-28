import typing

from globals import formats
from prompting.cli import helpers

#region Common
@helpers.register(str, 'Enter a string.')
@helpers.register(int, 'Enter an integer.')
@helpers.register(float, 'Enter a float.')
@helpers.register(complex, 'Enter a complex number.')
def prompt_for_str_castable(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a string castable.'
    ):
    return data_type(input(f'\n{title}\n{message}\n'))

@helpers.register(bool, 'Enter a boolean.')
def prompt_for_bool(data_type: type, title: str, message: str):
    return formats.BOOL_STRINGS[
        input(f'\n{title}\n{message}\n{formats.BOOL_STRINGS}\n')
    ]
#endregion

#region Collections
@helpers.register(set, 'Please enter a set of values.')
@helpers.register(list, 'Please enter a list of values.')
@helpers.register(tuple, 'Please enter a tuple of values.')
@helpers.register(frozenset, 'Please enter a frozenset of values.')
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
            temp_item_type: str = helpers.create_prompt(
                data_type=str, 
                title='Data Type', 
                message=f'Please input the type of element {len(output)}.'
            )
        input = helpers.create_prompt(list_item_type or temp_item_type)
        if input is None:
            break
        output.append(input)
    return data_type[output]

@helpers.register(dict, 'Please enter a dictionary of values.')
def prompt_for_dict(data_type: type, title: str, message: str):
    print(f'\n{title}\n{message}\n')
    args = typing.get_args(data_type)
    key_type = args[0] if args else str
    value_type = args[1] if len(args) > 1 else str
    output = {}
    while True:
        key = helpers.create_prompt(key_type, message='Please enter a key for a new pair.')
        if key is None:
            break
        value = helpers.create_prompt(value_type, message='Please enter a value for the pair.')
        if value is None:
            break
        output[key] = value
    return output
#endregion