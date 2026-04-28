import optuna.integration.pytorch_lightning
import lightning.pytorch.callbacks
import lightning.pytorch.profilers
import lightning.pytorch.loggers

from misc import logging_utils
from preprocessing import (
    datamodule,
    scalers
)
from . import (
    config,
    callbacks,
    helpers
)

logger = logging_utils.get_logger()

def objective(
        trial: optuna.Trial, 
        config: config.TrainConfig,
        data_module: datamodule.PurePlayDataModule,
        kill_callback: callbacks.KillTrainingCallback
    ) -> float:
    '''Objective function for hyperparameter tuning.'''
    # Choose a scaler
    scaler_name = config.scaler_name or trial.suggest_categorical(
        name='scaler_name',
        choices=[scaler.__name__ for scaler in scalers.SCALER_CACHE]
    )
    data_module.load(scaler_name)
    
    # Set up model
    model_params = helpers.suggest_model_params(trial, config)
    model = config.model_class(
        model_params=model_params, 
        data_params=data_module.params, 
        scaler=data_module.scaler
    )
    
    # Set up trainer
    trial_directory = f'{config.save_dir}/trial_{trial.number}'
    trainer = lightning.Trainer(
        max_epochs=-1,
        precision='16-mixed',
        #gradient_clip_val=1.0,
        #gradient_clip_algorithm='norm',
        callbacks=[
            kill_callback,
            lightning.pytorch.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.early_stopping_patience, 
                min_delta=config.early_stopping_delta,
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
                swa_lrs=model_params.learning_rate * config.swa_lr_factor or \
                model_params.learning_rate * trial.suggest_float(
                    name='swa_lr_factor',
                    low=config.swa_lr_factor_min,
                    high=config.swa_lr_factor_max
                ), 
                swa_epoch_start=config.swa_epoch_start or trial.suggest_int(
                    name='swa_epoch_start',
                    low=config.swa_epoch_start_min,
                    high=config.swa_epoch_start_max
                )
            ),
            lightning.pytorch.callbacks.LearningRateMonitor(
                logging_interval='epoch',
                log_momentum=True,
                log_weight_decay=True
            )
        ],
        profiler=lightning.pytorch.profilers.SimpleProfiler(
            dirpath=trial_directory, 
            filename='performance_log'
        ),
        logger=lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=config.save_dir, 
            version=f'trial_{trial.number}'
        )
    )

    # Train
    logger.info(f'\nTrial: {trial.number}')
    trainer.fit(model, datamodule=data_module)
    val_loss = trainer.callback_metrics.get('val_loss')
    return val_loss.item() if val_loss is not None else float('inf')