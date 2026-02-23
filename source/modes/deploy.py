import collections
import threading
import pyarrow
import queue
import torch
import time
import utilities, constants, devices, models, logger

def run_live_analysis(config: object) -> None:
    """Performs live analysis on input data using a pre-trained model."""
    model = models.BaseModel.load_model(config.model_file)
    kill_event = threading.Event()
    
    # Analysis Thread
    data_queue = queue.Queue()
    sequence_buffer = collections.deque(maxlen=2*model.data_params.polls_per_sequence)
    analysis_thread = threading.Thread(
        target=analysis_worker,
        args=(model, data_queue, kill_event),
        daemon=True
    )
    analysis_thread.start()

    # Parquet Writer Thread
    if config.write_to_file:
        file_name = f'{config.save_dir}/inputs_{time.strftime(constants.TIMESTAMP_FORMAT)}.parquet'
        schema = pyarrow.schema(
            fields=[(feature, pyarrow.float32()) for feature in model.data_params.whitelist],
            metadata={b'polling_rate': str(config.polling_rate).encode('utf-8')}
        )
        
        data_queue = queue.Queue()
        writer_thread = threading.Thread(
            target=utilities.parquet_writer_worker,
            args=(file_name, schema, data_queue, kill_event),
            daemon=True
        )
        writer_thread.start()

    # Mouse Listener Thread
    mouse_listener_thread = None
    if any(bind in config.mouse_whitelist for bind in constants.MOUSE_ANALOGS):
        mouse_listener_thread = threading.Thread(
            target=devices.listen_for_mouse_movement, 
            args=(kill_event,),
            daemon=True
        )
        mouse_listener_thread.start()

    poll_interval = 1.0 / config.polling_rate
    logger.info(f'Polling at {config.polling_rate}Hz (press {", ".join(config.kill_bind_list)} to stop)...')

    try:
        while True:
            if devices.should_kill(config):
                logger.info('Kill bind(s) detected. Stopping...')
                break

            row = devices.poll_if_capturing(config)
            if row:
                if config.write_to_file:
                    data_queue.put(row)
                sequence_buffer.append(row)
                if len(sequence_buffer) >= model.data_params.polls_per_sequence:
                    if config.deployment_window_type == constants.WindowType.TUMBLING:
                        data_queue.put(list(sequence_buffer)[:model.data_params.polls_per_sequence])
                        sequence_buffer.clear()
                    else: # sliding
                        data_queue.put(list(sequence_buffer)[-model.data_params.polls_per_sequence:])

            time.sleep(poll_interval)
    finally:
        kill_event.set()
        analysis_thread.join()
        if mouse_listener_thread:
            mouse_listener_thread.join()
        if config.write_to_file:
            writer_thread.join()
            logger.info(f'Data saved to {file_name}')

@torch.no_grad()
def analysis_worker(
        model: models.BaseModel,
        data_queue: queue.Queue,
        kill_event: threading.Event
    ) -> None:
    """Worker function to perform live analysis on input sequences."""
    while not kill_event.is_set():
        try:
            sequence = data_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        input_tensor = torch.tensor(sequence, dtype=torch.float32, device=devices.TORCH_DEVICE_TYPE)
        batched_input_tensor = input_tensor.unsqueeze(0)
        scaled_input_tensor = model.scale_data(batched_input_tensor)
        output = model(scaled_input_tensor)
        
        if model.training_type == constants.TrainingType.SUPERVISED:
            probabilities = torch.sigmoid(output)
            mean_confidence = probabilities.mean().item()
            logger.info(f'Confidence: {mean_confidence:.4f}')
        else:
            loss = model.loss_function(output, scaled_input_tensor)
            logger.info(f'Loss Function: {type(model.loss_function)}')
            logger.info(f'Reconstruction Error: {loss:.6f}')