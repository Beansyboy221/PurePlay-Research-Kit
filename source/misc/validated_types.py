import pydantic
import typing

from . import validators

ParquetPath = typing.Annotated[
    pydantic.FilePath, 
    validators.has_extension('.parquet')
]
CheckpointPath = typing.Annotated[
    pydantic.FilePath, 
    validators.has_extension('.ckpt')
]
GateCallable = typing.Callable[[typing.Iterable[bool]], bool]