import torch

from . import base


class DenseBinaryClassifier(base.ClassifierBase):
    def _define_layers(self):
        input_dimension = self.data_params.features_per_window

        layers = []
        current_dimension = input_dimension
        for _ in range(self.model_params.hidden_layers):
            layers.append(
                torch.nn.Linear(current_dimension, self.model_params.hidden_size)
            )
            layers.append(torch.nn.BatchNorm1d(self.model_params.hidden_size))
            layers.append(torch.nn.ELU())
            layers.append(torch.nn.Dropout(self.model_params.dropout))
            current_dimension = self.model_params.hidden_size
        layers = layers[:-3]
        layers.append(torch.nn.Linear(current_dimension, 1))
        self.model = torch.nn.Sequential(*layers).compile()

    def forward(self, input_window: torch.Tensor) -> torch.Tensor:
        flattened_input = input_window.flatten(start_dim=1)  # (Batch, window*Feature)
        logits: torch.Tensor = self.model(flattened_input)
        return logits.view(-1)
