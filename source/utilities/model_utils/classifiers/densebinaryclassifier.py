import torch

# PurePlay imports
from source.utilities.model_utils import modelregistry

# Model specific imports
from . import baseclassifier

@modelregistry.register()
class DenseBinaryClassifier(baseclassifier.ClassifierBase):
    def _define_layers(self):
        input_dimension = self.data_params.polls_per_sequence * self.data_params.features_per_poll
        
        layers = []
        current_dimension = input_dimension
        for _ in range(self.model_params.hidden_layers):
            layers.append(torch.nn.Linear(current_dimension, self.model_params.hidden_size))
            layers.append(torch.nn.BatchNorm1d(self.model_params.hidden_size))
            layers.append(torch.nn.ELU())
            layers.append(torch.nn.Dropout(self.model_params.dropout))
            current_dimension = self.model_params.hidden_size
        layers = layers[:-3]
        layers.append(torch.nn.Linear(current_dimension, 1))
        self.model = torch.nn.Sequential(*layers).compile()

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        flattened_input = input_sequence.flatten(start_dim=1) # (Batch, Sequence*Feature)
        logits = self.model(flattened_input)
        return logits.view(-1)
    


    