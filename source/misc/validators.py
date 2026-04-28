import functools
import pydantic
import inspect
import pathlib
import typing

def validate_extension(
        file_path: pathlib.Path, 
        expected_extension: str
    ) -> pathlib.Path:
    '''Validates a filepath based on its extension.'''
    if file_path.suffix.lower() != expected_extension.lower():
        raise ValueError(f"File must have a {expected_extension} extension")
    return file_path

def has_extension(ext: str | list[str]):
    return pydantic.AfterValidator(functools.partial(validate_extension, extensions=ext))

def validate_in_collection(
        thing: typing.Any,
        collection: typing.Collection
    ) -> typing.Any:
    '''Validates anything if it is in the given collection.'''
    if thing not in collection:
        raise ValueError(f'{thing} not found in set {collection}')
    return thing

def validate_is_concrete(thing: typing.Any) -> typing.Any:
    '''Validates anything if it is concrete (not abstract).'''
    if inspect.isabstract(thing):
        raise ValueError(f'{thing} is not concrete.')
    return thing