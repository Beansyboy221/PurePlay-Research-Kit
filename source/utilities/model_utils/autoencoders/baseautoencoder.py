import plotly.express
import pandas
import torch

# PurePlay imports
from source.utilities.model_utils import (
    modelparams,
    basemodel,
    enums
)
from source.utilities.data_utils import dataparams

class AutoencoderBase(basemodel.BaseModel):
    training_type = enums.TrainingType.UNSUPERVISED

    def __init__(
            self, 
            model_params: modelparams.ModelParams, 
            data_params: dataparams.DataParams, 
            scaler: object
        ):
        super().__init__(model_params, data_params, scaler)
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