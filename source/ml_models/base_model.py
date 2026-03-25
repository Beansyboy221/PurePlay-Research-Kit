import lightning
import sk2torch
import sklearn
import torch
import enum
import abc

from utilities.data_utils import dataparams
from utilities.mixins import on_init_mixin

from . import modelparams

class TrainingType(enum.StrEnum):
    SUPERVISED = enum.auto()
    UNSUPERVISED = enum.auto()

class BaseModel(
        lightning.LightningModule, 
        abc.ABC,
        on_init_mixin.OnInitMixin
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
            model_params: modelparams.ModelParams, 
            data_params: dataparams.ResolvedDataParams, 
            scaler: sklearn.base.TransformerMixin, # Could use better type checking.
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.model_params = model_params
        self.data_params = data_params
        self.scaler = sk2torch.wrap(scaler)

        self.save_hyperparameters({
            'model_class': self.__class__.__name__,
            'model_params': model_params.model_dump(), 
            'data_params': data_params.model_dump()
        })

        self._is_configured = False
        self.test_step_outputs = []

    def configure_optimizers(self) -> torch.optim.Optimizer | dict:
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
    def forward(self, input_window: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
