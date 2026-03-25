import torch

from . import baseclassifier

class LSTMBinaryClassifier(baseclassifier.ClassifierBase):
    def _define_layers(self):
        self.feature_extractor = torch.nn.LSTM(
            input_size=self.data_params.features_per_poll, 
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.hidden_layers,
            dropout=self.model_params.dropout 
            if self.model_params.hidden_layers > 1 else 0,
            batch_first=True
        ).compile()
        self.classifier = torch.nn.Linear(self.model_params.hidden_size, 1).compile()

    def forward(self, input_window: torch.Tensor) -> torch.Tensor:
        data, (hidden_state, cell_state) = self.feature_extractor(input_window)
        pooled_data = data.mean(dim=1) # Globally pool to get even coverage of the window
        logits = self.classifier(pooled_data)
        return logits.view(-1)
    