import tkinter.simpledialog
import tkinter.messagebox
import typing

from prompting.tkinter import helpers

#region Common
@helpers.register(str, 'Please select a string.')
def prompt_for_str(data_type: type, title: str, message: str):
    return tkinter.simpledialog.askstring(title, message)

@helpers.register(bool, 'Please select a boolean.')
def prompt_for_bool(data_type: type, title: str, message: str):
    return tkinter.messagebox.askyesno(title, message)
#endregion

#region Numerics
@helpers.register(int, 'Please enter an integer.')
def prompt_for_int(data_type: type, title: str, message: str):
    return tkinter.simpledialog.askinteger(title, message)

@helpers.register(float, 'Please enter a float.')
def prompt_for_float(data_type: type, title: str, message: str):
    return tkinter.simpledialog.askfloat(title, message)

@helpers.register(complex, 'Please enter a complex number.')
def prompt_for_complex(data_type: type, title: str, message: str):
    result = tkinter.simpledialog.askstring(title, message)
    try:
        return complex(result)
    except ValueError:
        tkinter.messagebox.showerror('Invalid Input.')
        return None
#endregion

#region Collections
@helpers.register(set, 'Please enter a set.')
@helpers.register(list, 'Please enter a list.')
@helpers.register(tuple, 'Please enter a tuple.')
@helpers.register(frozenset, 'Please enter an immutable set.')
def prompt_for_list(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a basic iterable.'
    ):
    args = typing.get_args(data_type)
    list_item_type = args[0] if args else str
    output = []
    while True:
        input = helpers.create_prompt(list_item_type)
        if input is None:
            break
        output.append(input)
    return data_type[output]

@helpers.register(dict, 'Please enter a dictionary.')
def prompt_for_dict(data_type: type, title: str, message: str):
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

#region HOW TO MAKE COLLECTIONS SUPPLY DEFAULT MESSAGES PROPERLY?