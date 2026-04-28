import torch

from . import base

class LSTMAutoencoder(base.AutoencoderBase):
    def _define_layers(self):
        self.encoder = torch.nn.LSTM(
            input_size=self.data_params.features_per_poll, 
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.hidden_layers,
            dropout=self.model_params.dropout 
            if self.model_params.hidden_layers > 1 else 0, 
            batch_first=True
        ).compile()
        self.compressor = torch.nn.Linear(
            self.model_params.hidden_size*2, 
            self.model_params.latent_size
        ).compile()
        self.decompressor = torch.nn.Linear(
            self.model_params.latent_size, 
            self.model_params.hidden_size
        ).compile()
        self.decoder = torch.nn.LSTM(
            input_size=self.model_params.latent_size, 
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.hidden_layers,
            dropout=self.model_params.dropout 
            if self.model_params.hidden_layers > 1 else 0, 
            batch_first=True
        ).compile()
        self.reconstructor = torch.nn.Linear(
            self.model_params.hidden_size, 
            self.data_params.features_per_poll
        ).compile()
    
    def forward(self, input_window: torch.Tensor) -> torch.Tensor:
        encoded_window, (hidden_state, cell_state) = self.encoder(input_window)
        final_states = torch.cat((hidden_state[-1], cell_state[-1]), dim=-1) 
        latent_vector: torch.Tensor = self.compressor(final_states)
        context_vector: torch.Tensor = self.decompressor(latent_vector) 
        repeat_vector = latent_vector.unsqueeze(1).repeat(1, self.data_params.polls_per_window, 1)
        final_states = context_vector.unsqueeze(0).repeat(self.model_params.hidden_layers, 1, 1)
        decoded_window, (hidden_state, cell_state) = self.decoder(repeat_vector, (final_states, final_states))
        return self.reconstructor(decoded_window)