import threading
import queue
import torch

# PurePlay imports
from globals.constants import processors
from models import (
    basemodel,
    enums
)
from source.globals import global_logger

@torch.no_grad()
def analysis_worker(
        model: basemodel.BaseModel,
        data_queue: queue.Queue,
        kill_event: threading.Event
    ) -> None:
    '''Worker function to perform live analysis on input sequences.'''
    while not kill_event.is_set():
        try:
            sequence = data_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        input_tensor = torch.tensor(
            data=sequence, 
            dtype=torch.float32, 
            device=processors.TORCH_DEVICE_TYPE
        ).unsqueeze(0) # Add a batch layer of size 1
        scaled_input_tensor = model.scale_data(input_tensor)
        output = model(scaled_input_tensor)
        
        match model.training_type:
            case enums.TrainingType.SUPERVISED:
                probabilities = torch.sigmoid(output)
                mean_confidence = probabilities.mean().item()
                global_logger.info(f'Confidence: {mean_confidence:.4f}')
            case enums.TrainingType.UNSUPERVISED:
                loss = model.loss_function(output, scaled_input_tensor)
                global_logger.info(f'Reconstruction Error: {loss:.6f}')
            case _:
                raise ValueError(f'Unsupported training type: {model.training_type}')