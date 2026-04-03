import pydantic
import sklearn
import torch

class ModelParams(pydantic.BaseModel):
    scaler: sklearn.base.TransformerMixin | None = pydantic.Field(
        default=None,
        description='The name of the scaler to use.',
        validation_alias=pydantic.AliasPath('train.tuning', 'scaler')
    )
    '''The name of the scaler to use.'''

    hidden_layers: int | None = pydantic.Field(
        default=None,
        ge=1,
        description='The number of hidden layers in the model.',
        validation_alias=pydantic.AliasPath('train.tuning', 'hidden_layers')
    )
    '''The number of hidden layers in the model.'''

    hidden_size: int | None = pydantic.Field(
        default=None,
        ge=1,
        description='The size of each/the first hidden layer in the model.',
        validation_alias=pydantic.AliasPath('train.tuning', 'base_hidden_size')
    )
    '''The size of each/the first hidden layer in the model.'''

    latent_size: int | None = pydantic.Field(
        default=None,
        ge=1,
        description='The size of the latent space in the model.',
        validation_alias=pydantic.AliasPath('train.tuning', 'latent_size')
    )
    '''The size of the latent space in the model.'''

    dropout: float | None = pydantic.Field(
        default=None, 
        ge=0.0,
        le=1.0,
        description='The dropout rate placed on each hidden layer.',
        validation_alias=pydantic.AliasPath('train.tuning', 'dropout')
    )
    '''The dropout rate placed on each hidden layer.'''

    optimizer: torch.optim.Optimizer | None = pydantic.Field(
        default=None,
        description='The name of the optimizer to use.',
        validation_alias=pydantic.AliasPath('train.tuning', 'optimizer')
    )
    '''The name of the optimizer to use.'''

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = pydantic.Field(
        default=None,
        description='The name of the scheduler to use.',
        validation_alias=pydantic.AliasPath('train.tuning', 'scheduler')
    )
    '''The name of the scheduler to use.'''

    learning_rate: float | None = pydantic.Field(
        default=None,
        gt=0.0,
        description='The base learning rate of the optimizer.',
        validation_alias=pydantic.AliasPath('train.tuning', 'learning_rate')
    )
    '''The base learning rate of the optimizer.'''

    weight_decay: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description='The weight decay applied to the optimizer.',
        validation_alias=pydantic.AliasPath('train.tuning', 'weight_decay')
    )
    '''The weight decay applied to the optimizer.'''

    momentum: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description='The momentum to use.',
        validation_alias=pydantic.AliasPath('train.tuning', 'momentum')
    )
    '''The momentum applied to the optimizer.'''