import pydantic

from preprocessing import scalers
from . import components

class ModelParams(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    
    scaler: scalers.SupportedScaler | None = pydantic.Field(default=None)
    '''The name of the scaler to use.'''

    hidden_layers: int | None = pydantic.Field(default=None, ge=1)
    '''The number of hidden layers in the model.'''

    hidden_size: int | None = pydantic.Field(default=None, ge=1)
    '''The size of each/the first hidden layer in the model.'''

    latent_size: int | None = pydantic.Field(default=None, ge=1)
    '''The size of the latent space in the model.'''

    dropout: float | None = pydantic.Field(default=None, ge=0.0, le=1.0)
    '''The dropout rate placed on each hidden layer.'''

    optimizer: components.SupportedOptimizer | None = pydantic.Field(default=None)
    '''The name of the optimizer to use.'''

    scheduler: components.SupportedScheduler | None = pydantic.Field(default=None)
    '''The name of the scheduler to use.'''

    learning_rate: float | None = pydantic.Field(default=None, gt=0.0)
    '''The base learning rate of the optimizer.'''

    weight_decay: float | None = pydantic.Field(default=None, ge=0.0, le=1.0)
    '''The weight decay applied to the optimizer.'''

    momentum: float | None = pydantic.Field(default=None, ge=0.0, le=1.0)
    '''The momentum applied to the optimizer.'''