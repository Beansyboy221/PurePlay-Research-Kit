import torch

from . import baseautoencoder

class CNNAutoencoder(baseautoencoder.AutoencoderBase):
    def _define_layers(self):
        input_channels = self.data_params.features_per_poll

        encoder_sizes = [input_channels]
        if self.model_params.hidden_layers == 1:
            encoder_sizes.append(self.model_params.latent_size)
        else:
            for i in range(self.model_params.hidden_layers):
                ratio = i / (self.model_params.hidden_layers - 1) # Ratio from 0.0 (hidden_size) to 1.0 (latent_size)
                layer_size = max(1, int(self.model_params.hidden_size * (self.model_params.latent_size / self.model_params.hidden_size) ** ratio))
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
                encoder_layers.append(torch.nn.Dropout1d(self.model_params.dropout))
        self.encoder = torch.nn.Sequential(*encoder_layers).compile()

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
        self.decoder = torch.nn.Sequential(*decoder_layers).compile()

    def forward(self, input_window: torch.Tensor) -> torch.Tensor:
        permuted_input = input_window.permute(0, 2, 1) # (Batch, Channel(Feature), window)
        encoded_window = self.encoder(permuted_input)
        decoded_window = self.decoder(encoded_window)
        return decoded_window.permute(0, 2, 1) # (Batch, window, Feature)