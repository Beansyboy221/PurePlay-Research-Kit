import pydantic
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

ParquetPath = typing.Annotated[
    pydantic.FilePath, 
    pydantic.AfterValidator(validate_extension('.parquet'))
]
CheckpointPath = typing.Annotated[
    pydantic.FilePath, 
    pydantic.AfterValidator(validate_extension('.ckpt'))
]