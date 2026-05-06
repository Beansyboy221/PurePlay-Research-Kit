import pydantic
import typing

from misc import validated_types, validators
import modes
from polling import params


class DeployConfig(modes.ModeConfig, params.PollParams, params.KillBindMixin):
    """Fields expected for deploy mode to work properly."""

    model_file: validated_types.CheckpointPath
    """The path to the model file."""

    write_to_file: bool = pydantic.Field(default=True)
    """Whether or not to write the data to a file."""

    save_dir: pydantic.DirectoryPath = pydantic.Field(default="data")
    """The directory to save the data to."""

    @pydantic.model_validator(mode="after")
    def validate_kill_binds(self) -> typing.Self:
        validators.validate_set_conflict(
            set_a=self.whitelist,
            set_b=self.kill_binds,
            msg="Whitelist cannot contain kill binds.",
        )
