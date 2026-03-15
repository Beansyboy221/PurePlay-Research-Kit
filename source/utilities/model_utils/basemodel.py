from abc import ABC as AbstractBaseClass
from abc import abstractmethod
import lightning
import typing
import numpy
import torch

# PurePlay imports
from source.globals.constants import tunables
from source.utilities.model_utils import (
    modelregistry,
    modelparams,
    enums
)
from source.utilities.data_utils import dataparams

#region Base Model
class BaseModel(lightning.LightningModule, AbstractBaseClass):
    training_type: enums.TrainingType | None = None

    def __init__(
            self, 
            model_params: modelparams.ModelParams, 
            data_params: dataparams.ResolvedDataParams, 
            scaler: object
        ):
        super().__init__()
        self.loss_function: torch.nn.Module | None = None
        self.model_params = model_params
        self.data_params = data_params
        self.scaler_name = type(scaler).__name__
        self.scaler_params = {}
        for attribute_name, value in vars(scaler).items():
            if attribute_name.endswith('_') and isinstance(value, numpy.ndarray):
                tensor_value = torch.from_numpy(value)
                self.scaler_params.update({attribute_name: tensor_value})
                if hasattr(self, attribute_name):
                    getattr(self, attribute_name).copy_(tensor_value)
                else:
                    self.register_buffer(attribute_name, tensor_value)
        
        # Saves hyperparameters and metadata to model.hparams
        self.save_hyperparameters({
            'model_class': self.__class__.__name__,
            'model_params': model_params.model_dump(), 
            'data_params': data_params.model_dump(),
            'scaler_name': self.scaler_name,
            'scaler_params': self.scaler_params
        })

        self.is_configured = False
        self.test_step_outputs = []

    def configure_model(self):
        if self.is_configured:
            return
        self._define_layers()
        self.is_configured = True

    @abstractmethod
    def _define_layers(self) -> None:
        '''Design your model structure here.'''
        pass

    @abstractmethod
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer | dict:
        '''Configures optimizers and schedulers based on model parameters.'''
        optimizer_class = tunables.OPTIMIZER_MAP[self.model_params.optimizer_name]
        optimizer_kwargs = {'lr': self.model_params.learning_rate}
        if self.model_params.weight_decay is not None:
            optimizer_kwargs['weight_decay'] = self.model_params.weight_decay
        if self.model_params.momentum is not None:
            optimizer_kwargs['momentum'] = self.model_params.momentum
        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

        if self.model_params.scheduler_name:
            scheduler_class = tunables.SCHEDULER_MAP[self.model_params.scheduler_name]
            scheduler = scheduler_class(optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                },
            }
        return optimizer

    def scale_data(self, input_sequence: torch.Tensor) -> torch.Tensor:
        '''
        Applies the correct scaling math dynamically based on the scaler type.
        Supports: StandardScaler, RobustScaler, MinMaxScaler, and MaxAbsScaler.
        '''
        scaler_operations = {
            'StandardScaler': lambda x, s: (x - s['mean_']) / s['scale_'],
            'RobustScaler': lambda x, s: (x - s['center_']) / s['scale_'],
            'MinMaxScaler': lambda x, s: x * s['scale_'] + s['min_'],
            'MaxAbsScaler': lambda x, s: x / s['scale_'],
        }
        if self.scaler_name not in scaler_operations:
            raise ValueError(f'Scaling logic for scaler: {self.scaler_name} is not implemented.')
        return scaler_operations[self.scaler_name](input_sequence, self.hparams.scaler_params)
        
    @classmethod
    def load_model(cls, checkpoint_path, **kwargs) -> typing.Self:
        '''Static method to load the correct child class automatically.'''
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        class_name = checkpoint.get('hyper_parameters').get('model_class')
        if not class_name:
            raise ValueError(f'Model file is missing model class name.')
        if class_name not in modelregistry.AVAILABLE_MODELS:
            raise ValueError(f'Unknown class: {class_name} in model file.')
        try:
            return modelregistry.AVAILABLE_MODELS[class_name].load_from_checkpoint(checkpoint_path, **kwargs)
        except:
            raise ValueError(f'Failed to load model of class: {class_name}.')
#endregion

