import pydantic

from beanml import config as model_config
from beaninput import config as poll_config
from beanapp import config as app_config

from . import tuning_overrides, search_spaces


class TrainConfig(
    app_config.StartupConfig,
    poll_config.PollConfig,
    model_config.ModelConfig,
    poll_config.KillBindMixin,
    search_spaces.SearchSpaces,
    tuning_overrides.TuningOverrides,
):
    """Fields expected for train mode to work properly."""

    save_dir: pydantic.DirectoryPath = pydantic.Field(default="models")
    """The directory to save the models to."""

    training_file_dir: pydantic.DirectoryPath = pydantic.Field(
        default="data\\train\\benign"
    )
    """The directory containing the training files."""

    validation_file_dir: pydantic.DirectoryPath = pydantic.Field(
        default="data\\val\\benign"
    )
    """The directory containing the validation files."""

    cheat_training_file_dir: pydantic.DirectoryPath | None = pydantic.Field(
        default="data\\train\\cheat"
    )
    """The directory containing the cheat training files."""

    cheat_validation_file_dir: pydantic.DirectoryPath | None = pydantic.Field(
        default="data\\val\\cheat"
    )
    """The directory containing the cheat validation files."""

    batch_size_tune_epochs: int = pydantic.Field(default=50, ge=1)
    """The number of epochs to tune the batch size for."""
