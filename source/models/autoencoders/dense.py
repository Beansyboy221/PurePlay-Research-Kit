import torch

from . import base

class DenseAutoencoder(base.AutoencoderBase):
    def _define_layers(self):
        input_dimension = self.data_params.polls_per_window * self.data_params.features_per_poll
        
        encoder_sizes = [input_dimension]
        if self.model_params.hidden_layers == 1:
            encoder_sizes.append(self.model_params.latent_size)
        else:
            for i in range(self.model_params.hidden_layers):
                ratio = i / (self.model_params.hidden_layers - 1) # Ratio from 0.0 (hidden_size) to 1.0 (latent_size)
                layer_size = max(1, int(self.model_params.hidden_size * (self.model_params.latent_size / self.model_params.hidden_size) ** ratio))
                encoder_sizes.append(layer_size)
        decoder_sizes = list(reversed(encoder_sizes))
        
        encoder_layers = []
        for i in range(len(encoder_sizes) - 1):
            encoder_layers.append(torch.nn.Linear(encoder_sizes[i], encoder_sizes[i+1]))
            if i < len(encoder_sizes) - 2:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_sizes[i+1]))
                encoder_layers.append(torch.nn.ELU())
                encoder_layers.append(torch.nn.Dropout(self.model_params.dropout))
        self.encoder = torch.nn.Sequential(*encoder_layers).compile()

        decoder_layers = []
        for i in range(len(decoder_sizes) - 1):
            decoder_layers.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
            if i < len(decoder_sizes) - 2:
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_sizes[i+1]))
                decoder_layers.append(torch.nn.ELU())
        self.decoder = torch.nn.Sequential(*decoder_layers).compile()

    def forward(self, input_window: torch.Tensor) -> torch.Tensor:
        flattened_input = input_window.flatten(start_dim=1)
        encoded_window = self.encoder(flattened_input)
        decoded_window: torch.Tensor = self.decoder(encoded_window)
        return decoded_window.unflatten(
            dim=1, 
            sizes=(self.data_params.polls_per_window, self.data_params.features_per_poll
        ))