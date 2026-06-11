import pydantic
import typing

from beaninput.config import PollConfig, KillBindMixin
from beanapp import StartupConfig
from beanml import CheckpointPath

from misc import validators


class DeployConfig(StartupConfig, PollConfig, KillBindMixin):
    """Fields expected for deploy mode to work properly."""

    model_file: CheckpointPath = pydantic.Field(
        description="The path to the model file."
    )
    """The path to the model file."""

    write_to_file: bool = pydantic.Field(
        default=True,
        description="Whether or not to write the data to a file.",
    )
    """Whether or not to write the data to a file."""

    save_dir: pydantic.DirectoryPath = pydantic.Field(
        default="data",
        description="The directory to save the data to.",
    )
    """The directory to save the data to."""

    @pydantic.model_validator(mode="after")
    def validate_kill_binds(self) -> typing.Self:
        validators.validate_set_conflict(
            set_a=self.whitelist,
            set_b=self.kill_binds,
            msg="Whitelist cannot contain kill binds.",
        )
