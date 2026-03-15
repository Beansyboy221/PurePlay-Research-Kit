from abc import ABC as AbstractBaseClass
import typing

class FileReader(AbstractBaseClass):
    extensions: tuple[str, ...]

    @classmethod
    def get_read_method(cls, output_type: type) -> typing.Callable | None:
        for method in cls.__dict__.values():
            if callable(method) and method.__annotations__.get('return') is output_type:
                return method
        return None