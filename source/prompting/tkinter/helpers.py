import functools
import typing

T = typing.TypeVar('T')

@functools.singledispatch
def create_prompt(
        data_type: T, 
        title: str = 'N/A', 
        message: str = ''
    ) -> T:
    '''Prompts the user for input using a Tkinter dialog.'''
    raise NotImplementedError(f'No TK implementation for type {data_type}')

def register(target_type, default_msg):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(data_type, title=None, message=None):
            msg = message if message is not None else default_msg
            return func(data_type, title, msg)
        create_prompt.register(target_type)(wrapper)
        return wrapper
    return decorator