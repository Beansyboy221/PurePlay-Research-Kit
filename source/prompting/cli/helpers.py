import functools
import typing

T = typing.TypeVar('T')

@functools.singledispatch
def create_prompt(
        data_type: T, 
        title: str = 'N/A', 
        message: str = ''
    ) -> T:
    '''
    Prompts the user for input using the command line interface.
    Using the same parameters as the other prompters allows for passing
    kwargs to the prompters without needing to know which one is being used.
    '''
    raise NotImplementedError(f'No CLI implementation for type {data_type}')

def register(target_type, default_msg='N/A'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(data_type, title=None, message=None):
            msg = message if message is not None else default_msg
            return func(data_type, title, msg)
        create_prompt.register(target_type)(wrapper)
        return wrapper
    return decorator