import pydantic

from ml_models import base_model

class TuningOverrides(pydantic.BaseModel):
    model_class: base_model.BaseModel | None = pydantic.Field(
        default=None,
        # validate that the model is concrete
        description='The model class name to use for training.',
        validation_alias=pydantic.AliasPath('train.overrides', 'model_class'),
    )
    '''The model class name to use for training.'''

    windows_per_batch: int | None = pydantic.Field(
        default=None,
        multiple_of=2,
        ge=16,
        description='The number of windows in each batch.',
        validation_alias=pydantic.AliasPath('train.overrides', 'windows_per_batch')
    )
    '''The number of windows in each batch.'''
    
    scaler_name: str | None = pydantic.Field(
        default=None,
        description='The name of the data scaler to use.',
        validation_alias=pydantic.AliasPath('train.overrides', 'scaler_name')
    )
    '''The name of the data scaler to use.'''

    early_stopping_patience: int = pydantic.Field(
        default=20,
        ge=1,
        description='The number of epochs to wait for improvement before stopping.',
        validation_alias=pydantic.AliasPath('train.overrides', 'early_stopping_patience')
    )
    '''The number of epochs to wait for improvement before stopping.'''

    early_stopping_delta: float = pydantic.Field(
        default=1e-4,
        ge=0.0,
        description='The minimum change in validation loss to qualify as improvement.',
        validation_alias=pydantic.AliasPath('train.overrides', 'early_stopping_delta')
    )
    '''The minimum change in validation loss to qualify as improvement.'''

    swa_epoch_start: int | None = pydantic.Field(
        default=None,
        description='The epoch at which stochastic weight averaging starts.',
        validation_alias=pydantic.AliasPath('train.overrides', 'swa_epoch_start')
    )
    '''The epoch at which stochastic weight averaging starts.'''

    swa_lr_factor: float | None = pydantic.Field(
        default=None,
        description='The factor by which the learning rate is multiplied.',
        validation_alias=pydantic.AliasPath('train.overrides', 'swa_lr_factor')
    )
    '''The factor by which the learning rate is multiplied.'''