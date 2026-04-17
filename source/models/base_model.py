import lightning
import sk2torch
import torch
import enum
import abc

from data.processing import (
    data_params,
    scalers
)
from misc import dunder_signals
from . import model_params

class TrainingType(enum.StrEnum):
    SUPERVISED = enum.auto()
    UNSUPERVISED = enum.auto()

class BaseModel(
        lightning.LightningModule, 
        abc.ABC,
        dunder_signals.OnInitSubclassMixin
    ):
    @property
    @abc.abstractmethod
    def training_type(self) -> TrainingType:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loss_function(self) -> torch.nn.Module:
        raise NotImplementedError

    def __init__(
            self, 
            model_params: model_params.ModelParams, 
            data_params: data_params.ResolvedDataParams, 
            scaler: scalers.SupportedScaler
        ):
        super().__init__()
        self.model_params = model_params
        self.data_params = data_params
        self.scaler = sk2torch.wrap(scaler)

        self.save_hyperparameters({
            'model_class': self.__class__.__name__,
            'model_params': model_params.model_dump(), 
            'data_params': data_params.model_dump(),
            'scaler_name': scaler.__class__.__name__
        })

        self._is_configured = False
        self.test_step_outputs = []

    def configure_optimizers(self):
        '''Configures optimizers and schedulers based on model parameters.'''
        optimizer_kwargs = {'lr': self.model_params.learning_rate}
        if self.model_params.weight_decay is not None:
            optimizer_kwargs['weight_decay'] = self.model_params.weight_decay
        if self.model_params.momentum is not None:
            optimizer_kwargs['momentum'] = self.model_params.momentum
        optimizer = self.model_params.optimizer(self.parameters(), **optimizer_kwargs)

        if self.model_params.scheduler:
            scheduler = self.model_params.scheduler(optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                },
            }
        return optimizer

    def configure_model(self):
        if self._is_configured:
            return
        self._define_layers()
        self._is_configured = True

    @abc.abstractmethod
    def _define_layers(self) -> None:
        '''Design your model structure here.'''
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward(self, input_window: torch.Tensor):
        raise NotImplementedError
