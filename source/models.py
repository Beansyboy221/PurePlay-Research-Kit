from abc import ABC as AbstractBaseClass
import plotly.express
import torchmetrics
import lightning
import pydantic
import typing
import pandas
import numpy
import torch
import preprocessing, constants

#region Runtime Model Registry
AVAILABLE_MODELS = {}

def register_model(model: type[lightning.LightningModule]) -> type[lightning.LightningModule]:
    """Decorator to register a model class."""
    AVAILABLE_MODELS[model.__name__] = model
    return model

def get_available_models() -> dict[str, type[lightning.LightningModule]]:
    """Returns a dictionary of available model classes."""
    return AVAILABLE_MODELS
#endregion

#region Metadata Configs
class ModelConfig(pydantic.BaseModel):
    """Parameters defining the neural network architecture and optimization."""
    hidden_layers: int = pydantic.Field(gt=0)
    hidden_size: int = pydantic.Field(gt=0)
    latent_size: int = pydantic.Field(gt=0)
    dropout: float = pydantic.Field(default=0.0, ge=0.0, le=0.5)
    optimizer_name: str = 'Adam'
    scheduler_name: typing.Optional[str] = None
    learning_rate: float = pydantic.Field(default=1e-3, gt=0)
    weight_decay: float = pydantic.Field(default=0, ge=0)
    momentum: float = pydantic.Field(default=0, ge=0)
#endregion

#region Base Models
class BaseModel(lightning.LightningModule, AbstractBaseClass):
    def __init__(self, model_params: ModelConfig, data_params: preprocessing.DataConfig, scaler_name: str = None):
        super().__init__()
        self.save_hyperparameters(ignore=['model_params', 'data_params'])
        self.hparams.update({
            'model_class': self.__class__.__name__,
            'model_params': model_params.model_dump(),
            'data_params': data_params.model_dump(),
            'scaler_params': {}
        })

        self.model_params = model_params
        self.data_params = data_params
        self.test_step_outputs = []

    def save_scaler(self, scaler) -> None:
        """Updates scaler params in model to match a scikit scaler."""
        self.hparams['scaler_name'] = type(scaler).__name__
        for attribute_name, value in vars(scaler).items():
            if attribute_name.endswith('_') and isinstance(value, numpy.ndarray):
                buffer_name = attribute_name
                tensor_value = torch.from_numpy(value).float()
                self.hparams['scaler_params'].update({buffer_name: tensor_value})
                if hasattr(self, buffer_name):
                    getattr(self, buffer_name).copy_(tensor_value)
                else:
                    self.register_buffer(buffer_name, tensor_value)

    def configure_optimizers(self) -> typing.Union[torch.optim.Optimizer, dict]:
        """Configures optimizers and schedulers based on model parameters."""
        optimizer_class = constants.OPTIMIZER_MAP[self.model_params.optimizer_name]
        optimizer_kwargs = {'lr': self.model_params.learning_rate}
        if self.model_params.weight_decay is not None:
            optimizer_kwargs['weight_decay'] = self.model_params.weight_decay
        if self.model_params.momentum is not None:
            optimizer_kwargs['momentum'] = self.model_params.momentum
        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

        if self.model_params.scheduler_name:
            scheduler_class = constants.SCHEDULER_MAP[self.model_params.scheduler_name]
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
        """
        Applies the correct scaling math dynamically based on the scaler type.
        Supports: StandardScaler, RobustScaler, MinMaxScaler, and MaxAbsScaler.
        """
        scaler_operations = {
            'StandardScaler': lambda x, s: (x - s['mean_']) / s['scale_'],
            'RobustScaler': lambda x, s: (x - s['center_']) / s['scale_'],
            'MinMaxScaler': lambda x, s: x * s['scale_'] + s['min_'],
            'MaxAbsScaler': lambda x, s: x / s['scale_'],
        }

        scaler_name = self.hparams['scaler_name']
        if not scaler_name:
            raise ValueError('Model has no scaler registered.')
        if scaler_name not in scaler_operations:
            raise NotImplementedError(f'Scaling logic for "{scaler_name}" is not implemented.')

        return scaler_operations[scaler_name](input_sequence, self.hparams['scaler_params'])
        
    @classmethod
    def load_model(cls, checkpoint_path, **kwargs):
        """Static method to load the correct child class automatically."""
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        hparams = checkpoint.get('hyper_parameters')
        if not hparams or 'model_class' not in hparams:
            raise ValueError(f'Model file is missing model class name.')
        class_name = hparams['model_class']
        if class_name not in AVAILABLE_MODELS:
            raise ValueError(f'Unknown class "{class_name}" in model file.')
        target_class = AVAILABLE_MODELS[class_name]
        try:
            return target_class.load_from_checkpoint(checkpoint_path, **kwargs)
        except Exception as e:
            raise ValueError(f'Failed to load model of class "{class_name}".') from e

class AutoencoderBase(BaseModel):
    training_type = constants.TrainingType.UNSUPERVISED

    def __init__(self, model_params: ModelConfig, data_params: preprocessing.DataConfig):
        super().__init__(model_params, data_params)
        self.loss_function = torch.nn.MSELoss()

    def _common_step(self, batch, batch_idx, stage: str) -> torch.Tensor:
        inputs, labels = batch
        reconstruction = self.forward(inputs)
        loss = self.loss_function(reconstruction, inputs)
        self.log(
            name=f'{stage}_loss', 
            value=loss, 
            on_step=False, 
            on_epoch=True
        )
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx) -> None:
        loss = self._common_step(batch, batch_idx, 'test')
        self.test_step_outputs.append({
            'Batch_Index': batch_idx, 
            'MSE_Loss': loss.detach().item()
    })

    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs:
            return
        figure = plotly.express.line(
            data_frame=pandas.DataFrame(self.test_step_outputs),
            x='Batch Index', 
            y='MSE Loss',
            title=f'{self.__class__.__name__} Reconstruction History:',
        )
        figure.write_html(f'{self.__class__.__name__}_reconstruction_history.html')
        self.test_step_outputs.clear()

class ClassifierBase(BaseModel):
    training_type = constants.TrainingType.SUPERVISED

    def __init__(self, model_params: ModelConfig, data_params: preprocessing.DataConfig):
        super().__init__(model_params, data_params)
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.test_accuracy = torchmetrics.BinaryAccuracy()
    
    def _common_step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core logic shared by train, val, and test."""
        inputs, labels = batch
        logits = self.forward(inputs)
        labels = labels.float().view_as(logits)
        loss = self.loss_function(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, labels = self._common_step(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, labels = self._common_step(batch)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx) -> None:
        loss, logits, labels = self._common_step(batch)
        self.test_accuracy(logits, labels)
        probability = torch.sigmoid(logits)
        self.log_dict({
            'test_loss': loss,
            'test_accuracy': self.test_accuracy,
            'test_mean_confidence': probability.mean()
        }, on_epoch=True)
        self.test_step_outputs.append({
            'Batch_Index': batch_idx,
            'Probability': probability.detach().cpu().item(),
            'True_Label': int(labels.detach().cpu().item())
        })

    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs:
            return
        figure = plotly.express.line(
            data_frame=pandas.DataFrame(self.test_step_outputs),
            x='Batch Index',
            y='Classification',
            title=f'{self.__class__.__name__} Classification History:',
        )
        figure.write_html(f'{self.__class__.__name__}_classification_history.html')
        self.test_step_outputs.clear()
#endregion

#region Dense Models
@register_model
class DenseAutoencoder(AutoencoderBase):
    def __init__(self, model_params: ModelConfig, data_params: preprocessing.DataConfig):
        super().__init__(model_params, data_params)
        input_dimension = data_params.polls_per_sequence * data_params.features_per_poll
        
        # Generate symmetrical cascading layer sizes
        encoder_sizes = [input_dimension]
        if model_params.hidden_layers == 1:
            encoder_sizes.append(model_params.latent_size)
        else:
            for i in range(model_params.hidden_layers):
                ratio = i / (model_params.hidden_layers - 1) # Ratio from 0.0 (hidden_size) to 1.0 (latent_size)
                layer_size = max(1, int(model_params.hidden_size * (model_params.latent_size / model_params.hidden_size) ** ratio))
                encoder_sizes.append(layer_size)
        decoder_sizes = list(reversed(encoder_sizes))
        
        encoder_layers = []
        for i in range(len(encoder_sizes) - 1):
            encoder_layers.append(torch.nn.Linear(encoder_sizes[i], encoder_sizes[i+1]))
            if i < len(encoder_sizes) - 2:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_sizes[i+1]))
                encoder_layers.append(torch.nn.ELU())
                encoder_layers.append(torch.nn.Dropout(model_params.dropout))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_sizes) - 1):
            decoder_layers.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
            if i < len(decoder_sizes) - 2:
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_sizes[i+1]))
                decoder_layers.append(torch.nn.ELU())
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        flattened_input = input_sequence.flatten(start_dim=1)
        encoded_sequence = self.encoder(flattened_input)
        decoded_sequence = self.decoder(encoded_sequence)
        return decoded_sequence.unflatten(
            dim=1, 
            sizes=(self.data_params.polls_per_sequence, self.data_params.features_per_poll
        ))

@register_model
class DenseBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: ModelConfig, data_params: preprocessing.DataConfig):
        super().__init__(model_params, data_params)
        input_dimension = data_params.polls_per_sequence * data_params.features_per_poll
        
        layers = []
        current_dimension = input_dimension
        for _ in range(model_params.hidden_layers):
            layers.append(torch.nn.Linear(current_dimension, model_params.hidden_size))
            layers.append(torch.nn.BatchNorm1d(model_params.hidden_size))
            layers.append(torch.nn.ELU())
            layers.append(torch.nn.Dropout(model_params.dropout))
            current_dimension = model_params.hidden_size
        layers = layers[:-3]
        layers.append(torch.nn.Linear(current_dimension, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        flattened_input = input_sequence.flatten(start_dim=1) # (Batch, Sequence*Feature)
        logits = self.layers(flattened_input)
        return logits.view(-1)
#endregion

#region 1D CNN Models
@register_model
class CNNAutoencoder(AutoencoderBase):
    def __init__(self, model_params: ModelConfig, data_params: preprocessing.DataConfig):
        super().__init__(model_params, data_params)
        input_channels = data_params.features_per_poll

        encoder_sizes = [input_channels]
        if model_params.hidden_layers == 1:
            encoder_sizes.append(model_params.latent_size)
        else:
            for i in range(model_params.hidden_layers):
                ratio = i / (model_params.hidden_layers - 1) # Ratio from 0.0 (hidden_size) to 1.0 (latent_size)
                layer_size = max(1, int(model_params.hidden_size * (model_params.latent_size / model_params.hidden_size) ** ratio))
                encoder_sizes.append(layer_size)
        encoder_layers = []
        for i in range(len(encoder_sizes) - 1):
            encoder_layers.append(torch.nn.Conv1d(
                in_channels=encoder_sizes[i], 
                out_channels=encoder_sizes[i+1], 
                kernel_size=3, 
                stride=2, 
                padding=1
            ))
            if i < len(encoder_sizes) - 2:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_sizes[i+1]))
                encoder_layers.append(torch.nn.ELU())
                encoder_layers.append(torch.nn.Dropout1d(model_params.dropout))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_sizes = list(reversed(encoder_sizes))
        decoder_layers = []
        for i in range(len(decoder_sizes) - 1):
            decoder_layers.append(torch.nn.Upsample(scale_factor=2)) # Scale factor must match encoder stride
            decoder_layers.append(torch.nn.Conv1d(
                in_channels=decoder_sizes[i], 
                out_channels=decoder_sizes[i+1], 
                kernel_size=3, 
                padding=1
            ))
            if i < len(decoder_sizes) - 2:
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_sizes[i+1]))
                decoder_layers.append(torch.nn.ELU())
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        permuted_input = input_sequence.permute(0, 2, 1) # (Batch, Channel(Feature), Sequence)
        encoded_sequence = self.encoder(permuted_input)
        decoded_sequence = self.decoder(encoded_sequence)
        return decoded_sequence.permute(0, 2, 1) # (Batch, Sequence, Feature)

@register_model
class CNNBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: ModelConfig, data_params: preprocessing.DataConfig):
        super().__init__(model_params, data_params)
        
        layers = []
        current_channels = data_params.features_per_poll
        for _ in range(model_params.hidden_layers):
            layers.append(torch.nn.Conv1d(current_channels, model_params.hidden_size, kernel_size=3))
            layers.append(torch.nn.BatchNorm1d(model_params.hidden_size))
            layers.append(torch.nn.ELU())
            layers.append(torch.nn.Dropout1d(model_params.dropout))
            current_channels = model_params.hidden_size
        layers = layers[:-3]
        self.feature_extractor = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(current_channels, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        permuted_input = input_sequence.permute(0, 2, 1) # (Batch, Channel(Feature), Sequence)
        extracted_features = self.feature_extractor(permuted_input)
        pooled_features = torch.mean(extracted_features, dim=2)
        logits = self.classifier(pooled_features)
        return logits.view(-1)
#endregion

#region GRU Models
@register_model
class GRUAutoencoder(AutoencoderBase):
    def __init__(self, model_params: ModelConfig, data_params: preprocessing.DataConfig):
        super().__init__(model_params, data_params)
        
        self.encoder = torch.nn.GRU(
            input_size=data_params.features_per_poll, 
            hidden_size=model_params.hidden_size,
            num_layers=model_params.hidden_layers,
            dropout=model_params.dropout if model_params.hidden_layers > 1 else 0, 
            batch_first=True
        )
        self.compressor = torch.nn.Linear(model_params.hidden_size, model_params.latent_size)
        self.decompressor = torch.nn.Linear(model_params.latent_size, model_params.hidden_size)
        self.decoder = torch.nn.GRU(
            input_size=model_params.latent_size, 
            hidden_size=model_params.hidden_size,
            num_layers=model_params.hidden_layers,
            dropout=model_params.dropout if model_params.hidden_layers > 1 else 0, 
            batch_first=True
        )
        self.reconstructor = torch.nn.Linear(model_params.hidden_size, data_params.features_per_poll)
    
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        encoded_sequence, hidden_state = self.encoder(input_sequence)
        latent_vector = self.compressor(hidden_state[-1])
        repeat_vector = latent_vector.unsqueeze(1).repeat(1, self.data_params.polls_per_sequence, 1)
        context_vector = self.decompressor(latent_vector) 
        context_vector = context_vector.unsqueeze(0).repeat(self.model_params.hidden_layers, 1, 1)
        decoded_sequence, hidden_state = self.decoder(repeat_vector, context_vector) # Am I passing too much?
        return self.reconstructor(decoded_sequence)

@register_model
class GRUBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: ModelConfig, data_params: preprocessing.DataConfig):
        super().__init__(model_params, data_params)

        self.feature_extractor = torch.nn.GRU(
            input_size=data_params.features_per_poll, 
            hidden_size=model_params.hidden_size,
            num_layers=model_params.hidden_layers,
            dropout=model_params.dropout if model_params.hidden_layers > 1 else 0,
            batch_first=True
        )
        self.classifier = torch.nn.Linear(model_params.hidden_size, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        data, hidden_state = self.feature_extractor(input_sequence)
        pooled_data = data.mean(dim=1) # Globally pool to get even coverage of the sequence
        logits = self.classifier(pooled_data)
        return logits.view(-1)
#endregion

#region LSTM Models
@register_model
class LSTMAutoencoder(AutoencoderBase):
    def __init__(self, model_params: ModelConfig, data_params: preprocessing.DataConfig):
        super().__init__(model_params, data_params)
        
        self.encoder = torch.nn.LSTM(
            input_size=data_params.features_per_poll, 
            hidden_size=model_params.hidden_size,
            num_layers=model_params.hidden_layers,
            dropout=model_params.dropout if model_params.hidden_layers > 1 else 0, 
            batch_first=True
        )
        self.compressor = torch.nn.Linear(model_params.hidden_size*2, model_params.latent_size)
        self.decompressor = torch.nn.Linear(model_params.latent_size, model_params.hidden_size)
        self.decoder = torch.nn.LSTM(
            input_size=model_params.latent_size, 
            hidden_size=model_params.hidden_size,
            num_layers=model_params.hidden_layers,
            dropout=model_params.dropout if model_params.hidden_layers > 1 else 0, 
            batch_first=True
        )
        self.reconstructor = torch.nn.Linear(model_params.hidden_size, data_params.features_per_poll)
    
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        encoded_sequence, (hidden_state, cell_state) = self.encoder(input_sequence)
        final_states = torch.cat((hidden_state[-1], cell_state[-1]), dim=-1) 
        latent_vector = self.compressor(final_states)
        context_vector = self.decompressor(latent_vector) 
        repeat_vector = latent_vector.unsqueeze(1).repeat(1, self.data_params.polls_per_sequence, 1)
        final_states = context_vector.unsqueeze(0).repeat(self.model_params.hidden_layers, 1, 1)
        decoded_sequence, (hidden_state, cell_state) = self.decoder(repeat_vector, (final_states, final_states))
        return self.reconstructor(decoded_sequence)

@register_model
class LSTMBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: ModelConfig, data_params: preprocessing.DataConfig):
        super().__init__(model_params, data_params)

        self.feature_extractor = torch.nn.LSTM(
            input_size=data_params.features_per_poll, 
            hidden_size=model_params.hidden_size,
            num_layers=model_params.hidden_layers,
            dropout=model_params.dropout if model_params.hidden_layers > 1 else 0,
            batch_first=True
        )
        self.classifier = torch.nn.Linear(model_params.hidden_size, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        data, (hidden_state, cell_state) = self.feature_extractor(input_sequence)
        pooled_data = data.mean(dim=1) # Globally pool to get even coverage of the sequence
        logits = self.classifier(pooled_data)
        return logits.view(-1)
#endregion