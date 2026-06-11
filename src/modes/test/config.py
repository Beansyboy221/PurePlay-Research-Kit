import pydantic

from beanapp import StartupConfig
from beanml import CheckpointPath


class TestConfig(StartupConfig):
    """Fields expected for test mode to work properly."""

    model_file: CheckpointPath = pydantic.Field(
        description="The path to the model file."
    )
    """The path to the model file."""

    save_dir: pydantic.DirectoryPath = pydantic.Field(
        default="reports",
        description="The directory to save the output reports to.",
    )
    """The directory to save the output reports to."""

    testing_file_dir: pydantic.DirectoryPath = pydantic.Field(
        default="data\\test",
        description="The directory containing the testing files.",
    )
    """The directory containing the testing files."""
