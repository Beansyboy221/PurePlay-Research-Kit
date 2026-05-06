import torchmetrics.classification
import plotly.express
import pandas
import torch
import abc

from models import base


class ClassifierBase(base.BaseModel, abc.ABC):
    training_type = base.TrainStrategy.SUPERVISED
    loss_function = torch.nn.BCEWithLogitsLoss()
    test_accuracy = torchmetrics.classification.BinaryAccuracy()

    def _common_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core logic shared by train, val, and test."""
        inputs, labels = batch
        logits = self.forward(inputs)
        labels = labels.float().view_as(logits)
        loss = self.loss_function(logits, labels)
        return loss, logits, labels

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loss, logits, labels = self._common_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loss, logits, labels = self._common_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, logits, labels = self._common_step(batch)
        self.test_accuracy(logits, labels)
        probability = torch.sigmoid(logits)
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": self.test_accuracy,
                "test_mean_confidence": probability.mean(),
            },
            on_epoch=True,
        )
        self.test_step_outputs.append(
            {
                "Batch_Index": batch_idx,
                "Probability": probability.detach().cpu().item(),
                "True_Label": int(labels.detach().cpu().item()),
            }
        )

    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs:
            return
        figure = plotly.express.line(
            data_frame=pandas.DataFrame(self.test_step_outputs),
            x="Batch Index",
            y="Classification",
            title=f"{self.__class__.__name__} Classification History:",
        )
        figure.write_html(f"{self.__class__.__name__}_classification_history.html")
        self.test_step_outputs.clear()
