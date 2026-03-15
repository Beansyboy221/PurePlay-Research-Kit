import optuna.integration.pytorch_lightning
import lightning.pytorch.callbacks
import lightning.pytorch.loggers
import lightning.pytorch.profilers

# PurePlay imports
from source.globals.constants import (
    searchspaces,
    tunables
)
from source.globals import (
    global_logger
)
from source.models import (
    modelparams,
    basemodel
)
from source.utilities import datamodule

# Mode-specific imports
from . import callbacks, helpers

def objective(
        trial: optuna.Trial, 
        model_class: type[basemodel.BaseModel], 
        data_module: datamodule.PurePlayDataModule,
        save_dir: str,
        manual_params: modelparams.ModelParams,
        kill_callback: callbacks.KillTrainingCallback
    ) -> float:
    '''Objective function for hyperparameter tuning.'''
    # Choose a scaler
    scaler_name = manual_params.scaler_name or trial.suggest_categorical(
        name='scaler_name',
        choices=list(tunables.SCALER_MAP.keys())
    )
    data_module.scaler_manager.load(scaler_name)
    
    # Set up model
    model_params = helpers.suggest_model_params(trial, data_module.data_params, manual_params)
    model = model_class(model_params, data_module.data_params, data_module.scaler_manager.scaler)
    
    # Set up trainer
    trial_directory = f'{save_dir}/trial_{trial.number}'
    trainer = lightning.Trainer(
        max_epochs=-1,
        precision='16-mixed',
        #gradient_clip_val=1.0,
        #gradient_clip_algorithm='norm',
        callbacks=[
            kill_callback,
            lightning.pytorch.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=searchspaces.EARLY_STOPPING_PATIENCE, 
                min_delta=searchspaces.EARLY_STOPPING_DELTA, 
                verbose=True
            ),
            lightning.pytorch.callbacks.ModelCheckpoint(
                monitor='val_loss', 
                dirpath=trial_directory
            ),
            optuna.integration.pytorch_lightning.PyTorchLightningPruningCallback(
                trial=trial, 
                monitor='val_loss'
            ),
            lightning.pytorch.callbacks.StochasticWeightAveraging(
                swa_lrs=model_params.learning_rate * manual_params.swa_lr_factor or \
                model_params.learning_rate * trial.suggest_float(
                    name='swa_lr_factor',
                    low=searchspaces.SWA_LR_FACTOR_MIN,
                    high=searchspaces.SWA_LR_FACTOR_MAX
                ), 
                swa_epoch_start=manual_params.swa_epoch_start or trial.suggest_int(
                    name='swa_epoch_start',
                    low=searchspaces.SWA_EPOCH_START_MIN,
                    high=searchspaces.SWA_EPOCH_START_MAX
                )
            ),
            lightning.pytorch.callbacks.LearningRateMonitor(
                logging_interval='epoch'
            )
        ],
        profiler=lightning.pytorch.profilers.SimpleProfiler(
            dirpath=trial_directory, 
            filename='performance_log'
        ),
        logger=lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=save_dir, 
            version=f'trial_{trial.number}'
        )
    )

    # Train
    global_logger.info(f'\nTrial: {trial.number}')
    trainer.fit(model, datamodule=data_module)
    val_loss = trainer.callback_metrics.get('val_loss')
    return val_loss.item() if val_loss is not None else float('inf')