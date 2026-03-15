import torch

# PurePlay imports
from source.utilities.model_utils import modelregistry

# Model specific imports
from . import baseclassifier

@modelregistry.register()
class CNNBinaryClassifier(baseclassifier.ClassifierBase):
    def _define_layers(self):
        layers = []
        current_channels = self.data_params.features_per_poll
        for _ in range(self.model_params.hidden_layers):
            layers.append(torch.nn.Conv1d(current_channels, self.model_params.hidden_size, kernel_size=3))
            layers.append(torch.nn.BatchNorm1d(self.model_params.hidden_size))
            layers.append(torch.nn.ELU())
            layers.append(torch.nn.Dropout1d(self.model_params.dropout))
            current_channels = self.model_params.hidden_size
        layers = layers[:-3]
        self.feature_extractor = torch.nn.Sequential(*layers).compile()
        self.classifier = torch.nn.Linear(current_channels, 1).compile()

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        permuted_input = input_sequence.permute(0, 2, 1) # (Batch, Channel(Feature), Sequence)
        extracted_features = self.feature_extractor(permuted_input)
        pooled_features = torch.mean(extracted_features, dim=2)
        logits = self.classifier(pooled_features)
        return logits.view(-1)