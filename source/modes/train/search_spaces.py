import pydantic

from models import (
    model_registry,
    base_model
)

class SearchSpaces(pydantic.BaseModel):
    model_classes: list[type[base_model.BaseModel]] = pydantic.Field(
        default_factory=model_registry.AVAILABLE_MODELS.values()
    )
    '''A list of model class names to search over.'''

    windows_per_batch_min: int = pydantic.Field(default=16, ge=1)
    '''The minimum number of windows in each batch.'''

    windows_per_batch_max: int = pydantic.Field(default=128, ge=1)
    '''The maximum number of windows in each batch.'''

    early_stopping_patience_min: int = pydantic.Field(default=5, ge=1)
    '''The minimum number of epochs to wait for improvement before stopping.'''

    early_stopping_patience_max: int = pydantic.Field(default=50, ge=1)
    '''The maximum number of epochs to wait for improvement before stopping.'''

    early_stopping_delta_min: float = pydantic.Field(default=1e-8, ge=0.0)
    '''The minimum change in validation loss to qualify as improvement.'''

    early_stopping_delta_max: float = pydantic.Field(default=1e-2, ge=0.0)
    '''The maximum change in validation loss to qualify as improvement.'''

    hidden_size_min: int = pydantic.Field(default=16, ge=1)
    '''The minimum number of units in the hidden layer.'''
    
    latent_size_min: int = pydantic.Field(default=8, ge=1)
    '''The minimum number of units in the latent (smallest) layer.'''

    hidden_layers_min: int = pydantic.Field(default=1, ge=1)
    '''The minimum number of hidden layers.'''

    hidden_layers_max: int = pydantic.Field(default=4, ge=1)
    '''The maximum number of hidden layers.'''

    dropout_min: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)
    '''The minimum dropout rate.'''

    dropout_max: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)
    '''The maximum dropout rate.'''

    learning_rate_min: float = pydantic.Field(default=1e-6, ge=1e-6, le=1e-2)
    '''The minimum learning rate.'''

    learning_rate_max: float = pydantic.Field(default=1e-3, ge=1e-6, le=1e-2)
    '''The maximum learning rate.'''

    weight_decay_min: float = pydantic.Field(default=1e-6, ge=1e-6, le=1e-2)
    '''The minimum weight decay.'''

    weight_decay_max: float = pydantic.Field(default=1e-3, ge=1e-6, le=1e-2)
    '''The maximum weight decay.'''

    momentum_min: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)
    '''The minimum momentum.'''

    momentum_max: float = pydantic.Field(default=0.99, ge=0.0, le=1.0)
    '''The maximum momentum.'''
    
    swa_epoch_start_min: int = pydantic.Field(default=20, ge=1)
    '''The minimum epoch at which stochastic weight averaging starts.'''

    swa_epoch_start_max: int = pydantic.Field(default=2000, ge=1)
    '''The maximum epoch at which stochastic weight averaging starts.'''

    swa_lr_factor_min: float = pydantic.Field(default=0.05, ge=0.0, le=1.0)
    '''The minimum factor by which the learning rate is multiplied.'''

    swa_lr_factor_max: float = pydantic.Field(default=0.8, ge=0.0, le=1.0)
    '''The maximum factor by which the learning rate is multiplied.'''