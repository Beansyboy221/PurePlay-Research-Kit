import threading
import queue
import torch

from globals import (
    processors,
    logger
)
from models import base_model

@torch.no_grad()
def analysis_worker(
        model: base_model.BaseModel,
        data_queue: queue.Queue,
        kill_event: threading.Event
    ) -> None:
    '''Worker function to perform live analysis on input windows.'''
    while not kill_event.is_set():
        try:
            window = data_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        input_tensor = torch.tensor(
            data=window, 
            dtype=torch.float32, 
            device=processors.TORCH_DEVICE_TYPE
        ).unsqueeze(0) # Add a batch layer of size 1
        scaled_input_tensor = model.scaler(input_tensor)
        output = model(scaled_input_tensor)
        
        match model.training_type:
            case base_model.TrainingType.SUPERVISED:
                probabilities = torch.sigmoid(output)
                mean_confidence = probabilities.mean().item()
                logger.info(f'Confidence: {mean_confidence:.4f}')
            case base_model.TrainingType.UNSUPERVISED:
                loss = model.loss_function(output, scaled_input_tensor)
                logger.info(f'Reconstruction Error: {loss:.6f}')