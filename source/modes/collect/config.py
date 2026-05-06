import pydantic
import typing

from misc import validators
import modes
from polling import params


class CollectConfig(modes.ModeConfig, params.PollParams, params.KillBindMixin):
    """Fields expected for collect mode to work properly."""

    save_dir: pydantic.DirectoryPath = pydantic.Field(default="data")
    """The directory to save the data to."""

    @pydantic.model_validator(mode="after")
    def validate_kill_binds(self) -> typing.Self:
        validators.validate_set_conflict(
            set_a=self.whitelist,
            set_b=self.kill_binds,
            msg="Whitelist cannot contain kill binds.",
        )
