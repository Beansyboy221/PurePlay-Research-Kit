import torch

from . import base


class GRUAutoencoder(base.AutoencoderBase):
    def _define_layers(self):
        self.encoder = torch.nn.GRU(
            input_size=self.data_params.features_per_poll,
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.hidden_layers,
            dropout=(
                self.model_params.dropout if self.model_params.hidden_layers > 1 else 0
            ),
            batch_first=True,
        ).compile()
        self.compressor = torch.nn.Linear(
            in_features=self.model_params.hidden_size,
            out_features=self.model_params.latent_size,
        ).compile()
        self.decompressor = torch.nn.Linear(
            in_features=self.model_params.latent_size,
            out_features=self.model_params.hidden_size,
        ).compile()
        self.decoder = torch.nn.GRU(
            input_size=self.model_params.latent_size,
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.hidden_layers,
            dropout=(
                self.model_params.dropout if self.model_params.hidden_layers > 1 else 0
            ),
            batch_first=True,
        ).compile()
        self.reconstructor = torch.nn.Linear(
            in_features=self.model_params.hidden_size,
            out_features=self.data_params.features_per_poll,
        ).compile()

    def forward(self, input_window: torch.Tensor) -> torch.Tensor:
        encoded_window, hidden_state = self.encoder(input_window)
        latent_vector: torch.Tensor = self.compressor(hidden_state[-1])
        repeat_vector = latent_vector.unsqueeze(1).repeat(
            1, self.data_params.polls_per_window, 1
        )
        context_vector: torch.Tensor = self.decompressor(latent_vector)
        context_vector = context_vector.unsqueeze(0).repeat(
            self.model_params.hidden_layers, 1, 1
        )
        decoded_window, hidden_state = self.decoder(
            repeat_vector, context_vector
        )  # Am I passing too much?
        return self.reconstructor(decoded_window)
