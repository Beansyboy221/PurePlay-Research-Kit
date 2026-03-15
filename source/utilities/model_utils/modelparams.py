import pydantic
import typing
import torch

# PurePlay imports
from source.globals.constants import searchspaces
from source.utilities.app_utils import config_utils

class ModelParams(pydantic.BaseModel):
    scaler_name: typing.Annotated[type[object], pydantic.BeforeValidator(config_utils.validate_scaler_name)] = pydantic.Field(
        default=None,
        description='The name of the scaler to use.',
        validation_alias=pydantic.AliasPath('train.tuning', 'scaler_name')
    )  
    '''The name of the scaler to use.'''

    hidden_layers: int | None = pydantic.Field(
        default=None,
        ge=searchspaces.HIDDEN_LAYERS_MIN,
        le=searchspaces.HIDDEN_LAYERS_MAX,
        description='The number of hidden layers in the model.',
        validation_alias=pydantic.AliasPath('train.tuning', 'hidden_layers')
    )
    '''The number of hidden layers in the model.'''

    hidden_size: int | None = pydantic.Field(
        default=None,
        ge=searchspaces.HIDDEN_SIZE_MIN,
        description='The size of each/the first hidden layer in the model.',
        validation_alias=pydantic.AliasPath('train.tuning', 'base_hidden_size')
    )
    '''The size of each/the first hidden layer in the model.'''

    latent_size: int | None = pydantic.Field(
        default=None,
        ge=searchspaces.LATENT_SIZE_MIN,
        description='The size of the latent space in the model.',
        validation_alias=pydantic.AliasPath('train.tuning', 'latent_size')
    )
    '''The size of the latent space in the model.'''

    dropout: float | None = pydantic.Field(
        default=None, 
        ge=searchspaces.DROPOUT_MIN, 
        le=searchspaces.DROPOUT_MAX, 
        description='The dropout rate placed on each hidden layer.',
        validation_alias=pydantic.AliasPath('train.tuning', 'dropout')
    )
    '''The dropout rate placed on each hidden layer.'''

    optimizer_name: typing.Annotated[type[torch.optim.Optimizer], pydantic.BeforeValidator(config_utils.validate_optimizer_name)] | None = pydantic.Field(
        default=None,
        description='The name of the optimizer to use.',
        validation_alias=pydantic.AliasPath('train.tuning', 'optimizer_name')
    )
    '''The name of the optimizer to use.'''

    scheduler_name: typing.Annotated[type[torch.optim.lr_scheduler._LRScheduler], pydantic.BeforeValidator(config_utils.validate_scheduler_name)] | None = pydantic.Field(
        default=None,
        description='The name of the scheduler to use.',
        validation_alias=pydantic.AliasPath('train.tuning', 'scheduler_name')
    )
    '''The name of the scheduler to use.'''

    learning_rate: float | None = pydantic.Field(
        default=None,
        gt=searchspaces.LEARNING_RATE_MIN,
        lt=searchspaces.LEARNING_RATE_MAX,
        description='The base learning rate of the optimizer.',
        validation_alias=pydantic.AliasPath('train.tuning', 'learning_rate')
    )
    '''The base learning rate of the optimizer.'''

    weight_decay: float | None = pydantic.Field(
        default=None,
        ge=searchspaces.WEIGHT_DECAY_MIN,
        le=searchspaces.WEIGHT_DECAY_MAX,
        description='The weight decay applied to the optimizer.',
        validation_alias=pydantic.AliasPath('train.tuning', 'weight_decay')
    )
    '''The weight decay applied to the optimizer.'''

    momentum: float | None = pydantic.Field(
        default=None,
        ge=searchspaces.MOMENTUM_MIN,
        le=searchspaces.MOMENTUM_MAX,
        description='The momentum to use.',
        validation_alias=pydantic.AliasPath('train.tuning', 'momentum')
    )
    '''The momentum applied to the optimizer.'''

    swa_epoch_start: int | None = pydantic.Field(
        default=None,
        ge=searchspaces.SWA_EPOCH_START_MIN,
        le=searchspaces.SWA_EPOCH_START_MAX,
        description='The epoch at which stochastic weight averaging starts.',
        validation_alias=pydantic.AliasPath('train.tuning', 'swa_epoch_start')
    )
    '''The epoch at which stochastic weight averaging starts.'''

    swa_lr_factor: float | None = pydantic.Field(
        default=None,
        ge=searchspaces.SWA_LR_FACTOR_MIN,
        le=searchspaces.SWA_LR_FACTOR_MAX,
        description='The factor by which the learning rate is multiplied.',
        validation_alias=pydantic.AliasPath('train.tuning', 'swa_lr_factor')
    )
    '''The factor by which the learning rate is multiplied.'''