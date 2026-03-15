import lightning.pytorch.tuner
import optuna

# PurePlay imports
from source.globals.constants import (
    searchspaces,
    tunables
)
from source.globals import global_logger
from source.utilities.model_utils import (
    basemodel,
    modelparams
)
from source.utilities.data_utils import (
    datamodule,
    dataparams
)

# Mode-specific imports
from . import callbacks

def tune_batch_size(
        model_class: type[basemodel.BaseModel], 
        data_module: datamodule.PurePlayDataModule,
        kill_callback: callbacks.KillTrainingCallback
    ) -> int:
    '''Automatically maximizes batch size for the worst-case model. Sets batch size in the data module.'''
    global_logger.info('Tuning batch size for hardware...')
    max_data_features = data_module.data_params.polls_per_sequence * len(data_module.data_params.whitelist)
    model_params = modelparams.ModelParams(
        hidden_layers=searchspaces.HIDDEN_LAYERS_MAX,
        hidden_size=max_data_features,
        latent_size=data_module.data_params.features_per_poll
    )
    model = model_class(model_params, data_module.data_params, data_module.scaler_manager.scaler)
    trainer = lightning.Trainer(
        max_epochs=searchspaces.BATCH_SIZE_TUNE_EPOCHS, 
        logger=False, 
        enable_model_summary=False, 
        callbacks=kill_callback
    )
    tuner = lightning.pytorch.tuner.Tuner(trainer)
    max_batch_size = tuner.scale_batch_size(model=model, datamodule=data_module) # This function also sets the self.batch_size property in the data module.
    if max_batch_size is None:
        global_logger.error('Batch size tuning failed to find a valid batch size.')
        raise RuntimeError("Batch size tuning failed to find a valid batch size.")
    global_logger.info(f'Maximum batch size found: {max_batch_size}')
    return max_batch_size

def suggest_model_params(
        trial: optuna.Trial, 
        data_params: dataparams.DataParams,
        manual_params: modelparams.ModelParams | None = None
    ) -> modelparams.ModelParams:
    """Uses optuna to pick a set of hyperparameters for the model."""
    max_data_features = data_params.polls_per_sequence * len(data_params.whitelist)
    hidden_size = manual_params.hidden_size or trial.suggest_int(
        name='hidden_size', 
        low=searchspaces.HIDDEN_SIZE_MIN, 
        high=max_data_features
    )

    optimizer_name = manual_params.optimizer_name or trial.suggest_categorical(
        name='optimizer_name', 
        choices=list(tunables.OPTIMIZER_MAP.keys())
    )
    scheduler_name = manual_params.scheduler_name or trial.suggest_categorical(
        name='scheduler_name', 
        choices=[None] + list(tunables.SCHEDULER_MAP.keys())
    )

    weight_decay = None
    if optimizer_name in tunables.OPTIMIZERS_WITH_WEIGHT_DECAY:
        weight_decay = manual_params.weight_decay or trial.suggest_float(
            name='weight_decay',
            low=searchspaces.WEIGHT_DECAY_MIN,
            high=searchspaces.WEIGHT_DECAY_MAX,
            log=True
        )
    momentum = None
    if optimizer_name in tunables.OPTIMIZERS_WITH_MOMENTUM:
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
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        learning_rate=manual_params.learning_rate or trial.suggest_float(
            name='learning_rate', 
            low=searchspaces.LEARNING_RATE_MIN, 
            high=searchspaces.LEARNING_RATE_MAX, 
            log=True
        ),
        weight_decay=weight_decay,
        momentum=momentum
    )
