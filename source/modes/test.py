import lightning
import preprocessing, models

def run_static_analysis(config: object) -> None:
    """Performs static analysis on selected data files using a pre-trained model."""
    model = models.BaseModel.load_model(config.model_file)
    data_module = preprocessing.PurePlayDataModule(config=config, data_params=model.data_params)
    data_module.update_scaler(model.hparams.scaler_name, model.hparams.scaler_params)
    trainer = lightning.Trainer(logger=False, enable_checkpointing=False)
    trainer.test(model, datamodule=data_module)
    # Graphing?
    # The output of the test method saves reconstruction history to a file.