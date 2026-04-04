import pydantic

from ml_models import base_model

class TuningOverrides(pydantic.BaseModel):
    model_class: base_model.BaseModel | None = pydantic.Field(
        default=None,
        # validate that the model is concrete
    )
    '''The model class name to use for training.'''

    windows_per_batch: int | None = pydantic.Field(
        default=None, 
        multiple_of=2,
        ge=16
    )
    '''The number of windows in each batch.'''
    
    scaler_name: str | None = pydantic.Field(default=None)
    '''The name of the data scaler to use.'''

    early_stopping_patience: int | None = pydantic.Field(default=None, ge=1)
    '''The number of epochs to wait for improvement before stopping.'''

    early_stopping_delta: float | None = pydantic.Field(default=None, ge=0.0)
    '''The minimum change in validation loss to qualify as improvement.'''

    swa_epoch_start: int | None = pydantic.Field(default=None)
    '''The epoch at which stochastic weight averaging starts.'''

    swa_lr_factor: float | None = pydantic.Field(default=None)
    '''The factor by which the learning rate is multiplied.'''