import optuna.integration.pytorch_lightning
import optuna.visualization
import lightning
import optuna
import torch
import time
import preprocessing, devices, constants, models, logger

def train_model(config: object) -> None:
    torch.serialization.add_safe_globals([models.ModelConfig, models.DataConfig])

    # Load and configure the data
    data_module = preprocessing.PurePlayDataModule(config)
    kill_callback = KillTrainingCallback(config)
    if config.sequences_per_batch:
        logger.info(f'Using batch size from config: {config.sequences_per_batch}')
        data_module.batch_size = config.sequences_per_batch
    else:
        _tune_batch_size(config, data_module, data_module.data_params, kill_callback)
        
    # Start optuna study
    study = optuna.create_study(study_name=f'{config.model_class}-{time.strftime(constants.TIMESTAMP_FORMAT)}')
    try:
        study.optimize(
            lambda trial: _objective(trial, config, data_module, data_module.data_params, kill_callback),
            callbacks=[kill_callback],
            gc_after_trial=True
        )
    except Exception as e:
        logger.error(f'Trial interrupted due to exception: {e}')
    if len(study.trials) == 0:
        logger.warning('No trials completed. Exiting early.')
        return
    
    # Generate study reports
    logger.info(f'\nBest trial: {study.best_trial.number}')
    logger.info(f'Best val loss: {study.best_value:.6f}')
    if len(study.trials) == 1:
        logger.warning('Not enough trials completed for a comparison. Skipping visualization.')
        return
    figure = optuna.visualization.plot_optimization_history(study)
    figure.write_html(f'{config.save_dir}/optimization_history.html')
    figure = optuna.visualization.plot_param_importances(study)
    figure.write_html(f'{config.save_dir}/param_importances.html')
    logger.info('Report graphs saved.')

#region Custom Callbacks
class KillTrainingCallback(lightning.pytorch.callbacks.Callback):
    """Kills current optuna study/trial upon pressing kill bind."""
    def __init__(self, config: object):
        super().__init__()
        self.config = config
        self.must_stop_study = False

    def on_train_batch_end(self, trainer, *args, **kwargs):
        """Pytorch Lightning hook to stop mid-trial."""
        if not self.must_stop_study and devices.should_kill(self.config):
            self.must_stop_study = True
            trainer.should_stop = True

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Optuna hook: Stops the study after the objective function returns."""
        if self.must_stop_study:
            study.stop()
#endregion

#region Objective Functions
def _objective(
        trial: optuna.Trial, 
        config: object, 
        data_module: preprocessing.PurePlayDataModule, 
        data_params: dict, 
        kill_callback: KillTrainingCallback
    ) -> float:
    """Objective function for hyperparameter tuning."""
    # Choose a scaler
    scaler_name = trial.suggest_categorical('scaler_name', list(constants.SCALER_MAP.keys()))
    data_module.update_scaler(scaler_name)
    
    # Set up model
    model_params = _suggest_model_params(trial, data_params)
    model = config.model_class(model_params, data_params)
    model.save_scaler(data_module.scaler)

    # Set up trainer
    swa_start = trial.suggest_int('swa_epoch_start', 20, 2000)
    swa_lr = model_params.learning_rate * trial.suggest_float('swa_lr_factor', 0.05, 0.8)
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
                patience=20, 
                min_delta=0.0001, 
                verbose=True
            ),
            lightning.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', dirpath=trial_directory),
            optuna.integration.pytorch_lightning.PyTorchLightningPruningCallback(trial, monitor='val_loss'),
            lightning.pytorch.callbacks.StochasticWeightAveraging(swa_lrs=swa_lr, swa_epoch_start=swa_start),
            lightning.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch')
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
#endregion

#region Helpers
def _tune_batch_size(
        config: object, 
        data_module: preprocessing.PurePlayDataModule,
        data_params: dict, 
        kill_callback: KillTrainingCallback
    ) -> int:
    """Automatically maximizes batch size for the worst-case model from the objective function. Sets batch size in the data module."""
    logger.info('Tuning batch size for hardware...')
    model_params = models.ModelConfig(
        hidden_layers=constants.MAX_HIDDEN_LAYERS,
        hidden_size=constants.MAX_HIDDEN_SIZE,
        latent_size=len(data_params.whitelist)
    )
    model = config.model_class(model_params, data_params)
    model.save_scaler(data_module.scaler)
    trainer = lightning.Trainer(
        max_epochs=50, 
        logger=False, 
        enable_model_summary=False, 
        callbacks=kill_callback
    )
    tuner = lightning.pytorch.tuner.Tuner(trainer)
    max_batch_size = tuner.scale_batch_size(model=model, datamodule=data_module) # This function also sets the self.batch_size property in the data module.
    logger.info(f'Maximum batch size found: {max_batch_size}')
    return max_batch_size

def _suggest_model_params(trial: optuna.Trial, data_params: preprocessing.DataConfig) -> dict:
    max_data_features = data_params.polls_per_sequence * len(data_params.whitelist)
    base_hidden_size = trial.suggest_int('hidden_size', 16, max_data_features)

    optimizer_name = trial.suggest_categorical('optimizer_name', list(constants.OPTIMIZER_MAP.keys()))
    scheduler_name = trial.suggest_categorical('scheduler_name', [None] + list(constants.SCHEDULER_MAP.keys()))
    weight_decay = None
    if optimizer_name in constants.OPTIMIZERS_WITH_WEIGHT_DECAY:
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    momentum = None
    if optimizer_name in constants.OPTIMIZERS_WITH_MOMENTUM:
        momentum = trial.suggest_float('momentum', 0.0, 0.99, log=True)

    return models.ModelConfig(
        hidden_layers=trial.suggest_int('hidden_layers', 1, constants.MAX_HIDDEN_LAYERS),
        hidden_size=base_hidden_size,
        latent_size=trial.suggest_int('latent_size', 8, base_hidden_size),
        dropout=trial.suggest_float('dropout', 0.0, 0.4, step=0.1),
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        learning_rate=trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
        weight_decay=weight_decay,
        momentum=momentum
    )
#endregion
