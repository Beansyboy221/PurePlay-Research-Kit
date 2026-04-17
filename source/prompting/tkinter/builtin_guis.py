import tkinter.simpledialog
import tkinter.messagebox
import typing

from . import tkinter_prompter

#region Common
@tkinter_prompter.register(str)
def prompt_for_str(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a string.'
    ):
    return tkinter.simpledialog.askstring(title, message)

@tkinter_prompter.register(bool)
def prompt_for_bool(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a boolean.'
    ):
    return tkinter.messagebox.askyesno(title, message)
#endregion

#region Numerics
@tkinter_prompter.register(int)
def prompt_for_int(
        data_type: type, 
        title: str, 
        message: str = 'Please enter an integer.'
    ):
    return tkinter.simpledialog.askinteger(title, message)

@tkinter_prompter.register(float)
def prompt_for_float(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a float.'
    ):
    return tkinter.simpledialog.askfloat(title, message)

@tkinter_prompter.register(complex)
def prompt_for_complex(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a complex number.'
    ):
    result = tkinter.simpledialog.askstring(title, message)
    try:
        return complex(result)
    except ValueError:
        tkinter.messagebox.showerror('Invalid Input.')
        return None
#endregion

#region Collections
@tkinter_prompter.register(set)
@tkinter_prompter.register(list)
@tkinter_prompter.register(tuple)
@tkinter_prompter.register(frozenset)
def prompt_for_list(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a list.'
    ):
    args = typing.get_args(data_type)
    list_item_type = args[0] if args else str
    output = []
    while True:
        input = tkinter_prompter.prompt_with_tkinter(list_item_type)
        if input is None:
            break
        output.append(input)
    return data_type[output]

@tkinter_prompter.register(dict)
def prompt_for_dict(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a dictionary.'
    ):
    args = typing.get_args(data_type)
    key_type = args[0] if args else str
    value_type = args[1] if len(args) > 1 else str
    output = {}
    while True:
        key = tkinter_prompter.prompt_with_tkinter(key_type, message='Please enter a key for a new pair.')
        if key is None:
            break
        value = tkinter_prompter.prompt_with_tkinter(value_type, message='Please enter a value for the pair.')
        if value is None:
            break
        output[key] = value
    return output
#endregion

#region HOW TO MAKE COLLECTIONS SUPPLY DEFAULT MESSAGES PROPERLY?