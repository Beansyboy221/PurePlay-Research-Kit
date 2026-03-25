import lightning.pytorch.tuner
import optuna

from globals.constants import searchspaces
from source.ml_models import base_model
from utilities.app_utils import global_logger
from source.ml_models import (
    modelparams
)
from utilities.data_utils import (
    datamodule,
    dataparams
)
from . import callbacks

def tune_batch_size(
        model_class: type[base_model.BaseModel], 
        data_module: datamodule.PurePlayDataModule,
        kill_callback: callbacks.KillTrainingCallback
    ) -> int:
    '''Automatically maximizes batch size for the worst-case model. Sets batch size in the data module.'''
    global_logger.info('Tuning batch size for hardware...')
    model_params = modelparams.ModelParams(
        hidden_layers=searchspaces.HIDDEN_LAYERS_MAX,
        hidden_size=data_module.data_params.features_per_window,
        latent_size=data_module.data_params.features_per_poll
    )
    model = model_class(
        model_params=model_params, 
        data_params=data_module.data_params, 
        scaler=data_module.scaler_manager.scaler
    )
    trainer = lightning.Trainer(
        max_epochs=searchspaces.BATCH_SIZE_TUNE_EPOCHS, 
        logger=False, 
        enable_model_summary=False, 
        callbacks=kill_callback
    )
    tuner = lightning.pytorch.tuner.Tuner(trainer)
    max_batch_size = tuner.scale_batch_size(model=model, datamodule=data_module)
    if max_batch_size is None:
        raise RuntimeError('Batch size tuning failed to find a valid batch size.')
    global_logger.info(f'Maximum batch size found: {max_batch_size}')
    return max_batch_size

def suggest_model_params(
        trial: optuna.Trial, 
        data_params: dataparams.DataParams,
        manual_params: modelparams.ModelParams | None = None
    ) -> modelparams.ModelParams:
    '''Uses optuna to pick a set of hyperparameters for the model.'''
    hidden_size = manual_params.hidden_size or trial.suggest_int(
        name='hidden_size',
        low=searchspaces.HIDDEN_SIZE_MIN, 
        high=data_params.features_per_window
    )

    optimizer = manual_params.optimizer or trial.suggest_categorical(
        name='optimizer', 
        choices=searchspaces.SUPPORTED_OPTIMIZERS
    )
    scheduler = manual_params.scheduler or trial.suggest_categorical(
        name='scheduler', 
        choices=[None] + searchspaces.SUPPORTED_SCHEDULERS
    )

    weight_decay = None
    if hasattr(optimizer, 'weight_decay'):
        weight_decay = manual_params.weight_decay or trial.suggest_float(
            name='weight_decay',
            low=searchspaces.WEIGHT_DECAY_MIN,
            high=searchspaces.WEIGHT_DECAY_MAX,
            log=True
        )
    
    momentum = None
    if hasattr(optimizer, 'momentum'):
        momentum = manual_params.momentum or trial.suggest_float(
            name='momentum',
            low=searchspaces.MOMENTUM_MIN,
            high=searchspaces.MOMENTUM_MAX
        )

    return modelparams.ModelParams(
        hidden_layers=manual_params.hidden_layers or trial.suggest_int(   
            name='hidden_layers', 
            low=searchspaces.HIDDEN_LAYERS_MIN, 
            high=searchspaces.HIDDEN_LAYERS_MAX
        ),
        hidden_size=hidden_size,
        latent_size=manual_params.latent_size or trial.suggest_int(
            name='latent_size', 
            low=searchspaces.LATENT_SIZE_MIN, 
            high=hidden_size
        ),
        dropout=manual_params.dropout or trial.suggest_float(
            name='dropout', 
            low=searchspaces.DROPOUT_MIN, 
            high=searchspaces.DROPOUT_MAX
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        learning_rate=manual_params.learning_rate or trial.suggest_float(
            name='learning_rate', 
            low=searchspaces.LEARNING_RATE_MIN, 
            high=searchspaces.LEARNING_RATE_MAX, 
            log=True
        ),
        weight_decay=weight_decay,
        momentum=momentum
    )
