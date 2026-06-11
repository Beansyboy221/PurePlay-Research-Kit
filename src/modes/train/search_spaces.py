import pydantic

import beanml


class SearchSpaces(pydantic.BaseModel):
    model_classes: list[type[beanml.BaseModel]] = pydantic.Field(
        default_factory=beanml.AVAILABLE_MODELS.values(),
        description="A list of model class names to search over.",
    )
    """A list of model class names to search over."""

    windows_per_batch_min: int = pydantic.Field(
        default=16,
        description="The minimum number of windows in each batch.",
        ge=1,
    )
    """The minimum number of windows in each batch."""

    windows_per_batch_max: int = pydantic.Field(
        default=128,
        description="The maximum number of windows in each batch.",
        ge=1,
    )
    """The maximum number of windows in each batch."""

    early_stopping_patience_min: int = pydantic.Field(
        default=5,
        description="The minimum number of epochs to wait for improvement before stopping.",
        ge=1,
    )
    """The minimum number of epochs to wait for improvement before stopping."""

    early_stopping_patience_max: int = pydantic.Field(
        default=50,
        description="The maximum number of epochs to wait for improvement before stopping.",
        ge=1,
    )
    """The maximum number of epochs to wait for improvement before stopping."""

    early_stopping_delta_min: float = pydantic.Field(
        default=1e-8,
        description="The minimum change in validation loss to qualify as improvement.",
        ge=0.0,
    )
    """The minimum change in validation loss to qualify as improvement."""

    early_stopping_delta_max: float = pydantic.Field(
        default=1e-2,
        description="The maximum change in validation loss to qualify as improvement.",
        ge=0.0,
    )
    """The maximum change in validation loss to qualify as improvement."""

    hidden_size_min: int = pydantic.Field(
        default=16,
        description="The minimum number of units in the hidden layer.",
        ge=1,
    )
    """The minimum number of units in the hidden layer."""

    latent_size_min: int = pydantic.Field(
        default=8,
        description="The minimum number of units in the latent (smallest) layer.",
        ge=1,
    )
    """The minimum number of units in the latent (smallest) layer."""

    hidden_layers_min: int = pydantic.Field(
        default=1,
        description="The minimum number of hidden layers.",
        ge=1,
    )
    """The minimum number of hidden layers."""

    hidden_layers_max: int = pydantic.Field(
        default=4,
        description="The maximum number of hidden layers.",
        ge=1,
    )
    """The maximum number of hidden layers."""

    dropout_min: float = pydantic.Field(
        default=0.0,
        description="The minimum dropout rate.",
        ge=0.0,
        le=1.0,
    )
    """The minimum dropout rate."""

    dropout_max: float = pydantic.Field(
        default=0.5,
        description="The maximum dropout rate.",
        ge=0.0,
        le=1.0,
    )
    """The maximum dropout rate."""

    learning_rate_min: float = pydantic.Field(
        default=1e-6,
        description="The minimum learning rate.",
        ge=1e-6,
        le=1e-2,
    )
    """The minimum learning rate."""

    learning_rate_max: float = pydantic.Field(
        default=1e-3,
        description="The maximum learning rate.",
        ge=1e-6,
        le=1e-2,
    )
    """The maximum learning rate."""

    weight_decay_min: float = pydantic.Field(
        default=1e-6,
        description="The minimum weight decay.",
        ge=1e-6,
        le=1e-2,
    )
    """The minimum weight decay."""

    weight_decay_max: float = pydantic.Field(
        default=1e-3,
        description="The maximum weight decay.",
        ge=1e-6,
        le=1e-2,
    )
    """The maximum weight decay."""

    momentum_min: float = pydantic.Field(
        default=0.0,
        description="The minimum momentum.",
        ge=0.0,
        le=1.0,
    )
    """The minimum momentum."""

    momentum_max: float = pydantic.Field(
        default=0.99,
        description="The maximum momentum.",
        ge=0.0,
        le=1.0,
    )
    """The maximum momentum."""

    swa_epoch_start_min: int = pydantic.Field(
        default=20,
        description="The minimum epoch at which stochastic weight averaging starts.",
        ge=1,
    )
    """The minimum epoch at which stochastic weight averaging starts."""

    swa_epoch_start_max: int = pydantic.Field(
        default=2000,
        description="The maximum epoch at which stochastic weight averaging starts.",
        ge=1,
    )
    """The maximum epoch at which stochastic weight averaging starts."""

    swa_lr_factor_min: float = pydantic.Field(
        default=0.05,
        description="The minimum factor by which the learning rate is multiplied.",
        ge=0.0,
        le=1.0,
    )
    """The minimum factor by which the learning rate is multiplied."""

    swa_lr_factor_max: float = pydantic.Field(
        default=0.8,
        description="The maximum factor by which the learning rate is multiplied.",
        ge=0.0,
        le=1.0,
    )
    """The maximum factor by which the learning rate is multiplied."""
