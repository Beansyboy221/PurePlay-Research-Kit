import functools
import importlib
import pkgutil
import typing

T = typing.TypeVar('T')

@functools.singledispatch
def prompt_with_tkinter(
        data_type: T, 
        title: str = 'N/A', 
        message: str = ''
    ) -> T:
    '''Prompts the user for input using a Tkinter dialog.'''
    raise NotImplementedError(f'No TK implementation for type {data_type}')

def register(target_type, default_msg):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(data_type, title="Input", message=None):
            msg = message if message is not None else default_msg
            return func(data_type, title, msg)
        register(target_type)(wrapper)
        return wrapper
    return decorator

# Prompting with TK operates through indirect recursion
# (each value needed in a collection gets its own window)

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, __name__ + '.'):
    importlib.import_module(module_name)