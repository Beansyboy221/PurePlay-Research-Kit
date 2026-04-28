'''A dynamic user prompting system.'''
import typing

PromptMethod = typing.Callable[[type, str, str], typing.Any]
