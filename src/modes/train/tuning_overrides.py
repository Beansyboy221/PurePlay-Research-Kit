import typing

import pydantic

import beanml

from misc import validators


class TuningOverrides(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model_class: beanml.BaseModel | None = pydantic.Field(
        default=None,
        description="The model class name to use for training.",
    )
    """The model class name to use for training."""

    windows_per_batch: int | None = pydantic.Field(
        default=None,
        description="The number of windows in each batch.",
        multiple_of=2,
        ge=16,
    )
    """The number of windows in each batch."""

    scaler_name: str | None = pydantic.Field(
        default=None, description="The name of the data scaler to use."
    )
    """The name of the data scaler to use."""

    early_stopping_patience: int | None = pydantic.Field(
        default=None,
        description="The number of epochs to wait for improvement before stopping.",
        ge=1,
    )
    """The number of epochs to wait for improvement before stopping."""

    early_stopping_delta: float | None = pydantic.Field(
        default=None,
        description="The minimum change in validation loss to qualify as improvement.",
        ge=0.0,
    )
    """The minimum change in validation loss to qualify as improvement."""

    swa_epoch_start: int | None = pydantic.Field(
        default=None,
        description="The epoch at which stochastic weight averaging starts.",
    )
    """The epoch at which stochastic weight averaging starts."""

    swa_lr_factor: float | None = pydantic.Field(
        default=None, description="The factor by which the learning rate is multiplied."
    )
    """The factor by which the learning rate is multiplied."""

    @pydantic.model_validator(mode="after")
    def validate_concrete_model(self) -> typing.Self:
        validators.validate_is_concrete(thing=self.model_class)
