import lightning
import logging

from misc import cuda_helpers
from models import model_registry
from data.processing import (
    datamodule,
    scalers
)
from . import test_config

def test(config: test_config.TestConfig) -> None:
    '''
    Main entry point for test mode.
    Performs static analysis on selected data files using a pre-trained model.
    '''
    cuda_helpers.optimize_cuda_for_hardware()
    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)

    model = model_registry.load_model(config.model_file)
    scaler_class: scalers.SupportedScaler | None = None
    for scaler_class in scalers.SCALER_CACHE:
        if scaler_class.__name__ == model.hparams.scaler_name:
            scaler_class = scaler_class
    if not scaler_class:
        raise RuntimeError('Model is missing a recognized scaler name.')
    
    data_module = datamodule.PurePlayDataModule(
        data_params=model.data_params,
        testing_dir=config.testing_file_dir
    )
    data_module.scaler_manager.load(scaler_class)

    trainer = lightning.Trainer(logger=False, enable_checkpointing=False)
    trainer.test(model=model, datamodule=data_module)
    # Graphing?
    # The output of the test method currently 
    # saves reconstruction history to a file.