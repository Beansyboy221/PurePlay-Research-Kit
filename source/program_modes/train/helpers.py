import lightning.pytorch.tuner
import optuna

from ml_models import (
    model_params,
    components
)
from utilities.app_utils import global_logger
from utilities.data_utils import datamodule
from . import (
    callbacks,
    train_config
)

def tune_batch_size(
        config: train_config.TrainConfig,
        data_module: datamodule.PurePlayDataModule,
        kill_callback: callbacks.KillTrainingCallback
    ) -> int:
    '''Automatically maximizes batch size for the worst-case model. Sets batch size in the data module.'''
    global_logger.info('Tuning batch size for hardware...')
    model = config.model_class(
        model_params=model_params.ModelParams(
            hidden_layers=config.hidden_layers_max,
            hidden_size=data_module.data_params.features_per_window,
            latent_size=data_module.data_params.features_per_poll
        ),
        data_params=data_module.data_params, 
        scaler=data_module.scaler_manager.scaler
    )
    trainer = lightning.Trainer(
        max_epochs=config.batch_size_tune_epochs, 
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
        config: train_config.TrainConfig,
    ) -> model_params.ModelParams:
    '''Uses optuna to pick a set of hyperparameters for the model.'''
    hidden_size = config.hidden_size or trial.suggest_int(
        name='hidden_size',
        low=config.hidden_size_min, 
        high=config.features_per_window # Why isn't this highlighting?
    )

    optimizer = config.optimizer or trial.suggest_categorical(
        name='optimizer', 
        choices=components.SUPPORTED_OPTIMIZERS
    )
    scheduler = config.scheduler or trial.suggest_categorical(
        name='scheduler', 
        choices=[None] + components.SUPPORTED_SCHEDULERS
    )

    weight_decay = None
    if hasattr(optimizer, 'weight_decay'):
        weight_decay = config.weight_decay or trial.suggest_float(
            name='weight_decay',
            low=config.weight_decay_min,
            high=config.weight_decay_max,
            log=True
        )
    
    momentum = None
    if hasattr(optimizer, 'momentum'):
        momentum = config.momentum or trial.suggest_float(
            name='momentum',
            low=config.momentum_min,
            high=config.momentum_max
        )

    return model_params.ModelParams(
        hidden_layers=config.hidden_layers or trial.suggest_int(   
            name='hidden_layers', 
            low=config.hidden_layers_min, 
            high=config.hidden_layers_max
        ),
        hidden_size=hidden_size,
        latent_size=config.latent_size or trial.suggest_int(
            name='latent_size', 
            low=config.latent_size_min, 
            high=hidden_size
        ),
        dropout=config.dropout or trial.suggest_float(
            name='dropout', 
            low=config.dropout_min, 
            high=config.dropout_max
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        learning_rate=config.learning_rate or trial.suggest_float(
            name='learning_rate', 
            low=config.learning_rate_min, 
            high=config.learning_rate_max, 
            log=True
        ),
        weight_decay=weight_decay,
        momentum=momentum
    )
