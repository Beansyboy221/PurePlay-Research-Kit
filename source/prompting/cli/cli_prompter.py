import importlib
import functools
import pkgutil
import typing

T = typing.TypeVar('T')

@functools.singledispatch
def prompt_with_cli(
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

def register_prompt(target_type, default_msg):
    def decorator(func):
        # We wrap the function to inject the default message if None is passed
        @functools.wraps(func)
        def wrapper(data_type, title="Input", message=None):
            msg = message if message is not None else default_msg
            return func(data_type, title, msg)
        prompt_with_cli.register(target_type)(wrapper)
        return wrapper
    return decorator

# Should we differentiate between selecting and building the type in params?
# Should this instead be handled by the type (use an enum type)?

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, __name__ + '.'):
    importlib.import_module(module_name)