"""Common custom data types."""

import typing

import pydantic

from . import validators

TOMLPath = typing.Annotated[pydantic.FilePath, validators.has_extension(".toml")]
ParquetPath = typing.Annotated[pydantic.FilePath, validators.has_extension(".parquet")]
