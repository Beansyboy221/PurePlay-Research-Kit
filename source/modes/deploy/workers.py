import threading
import sk2torch
import queue
import torch

from globals import processors
from models import base


@torch.no_grad()
def analysis_worker(
    model: base.BaseModel, data_queue: queue.Queue, kill_event: threading.Event
) -> None:
    """Worker function to perform live analysis on input windows."""
    scaler_module = sk2torch.wrap(model.scaler)
    while not kill_event.is_set():
        try:
            window = data_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        input_tensor = torch.tensor(
            data=window, dtype=torch.float32, device=processors.TORCH_DEVICE_TYPE
        ).unsqueeze(
            0
        )  # Add a batch layer of size 1
        scaled_input_tensor = scaler_module(input_tensor)
        output = model(scaled_input_tensor)

        match model.train_strategy:
            case base.TrainStrategy.SUPERVISED:
                probabilities = torch.sigmoid(output)
                mean_confidence = probabilities.mean().item()
                # logger.info(f"Confidence: {mean_confidence:.4f}")
            case base.TrainStrategy.UNSUPERVISED:
                loss = model.loss_function(output, scaled_input_tensor)
                # logger.info(f"Reconstruction Error: {loss:.6f}")
