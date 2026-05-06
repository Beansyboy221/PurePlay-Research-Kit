import pydantic

import modes
from models import params as model_params
from polling import params as poll_params
from . import tuning_overrides, search_spaces


class TrainConfig(
    modes.ModeConfig,
    poll_params.PollParams,
    model_params.ModelParams,
    poll_params.KillBindMixin,
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
