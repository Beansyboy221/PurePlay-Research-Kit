import collections
import threading
import pyarrow
import logging
import queue
import time

from data import writers
from data.polling import (
    windows_raw_mouse,
    poll_helpers,
    bind_enums
)
from globals import (
    formats,
    logger
)
import misc.cuda_helpers as cuda_helpers
from models import model_registry
from . import (
    deploy_config,
    workers
)

def deploy(config: deploy_config.DeployConfig) -> None:
    '''
    Main entry point for deployment mode.
    Performs live analysis on input data using a pre-trained model.
    '''
    cuda_helpers.optimize_cuda_for_hardware()
    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    model = model_registry.load_model(config.model_file)
    config.ignore_empty_polls = model.data_params.ignore_empty_polls
    kill_event = threading.Event()
    
    # Analysis Thread
    analysis_queue = queue.Queue()
    window_ring_buffer = collections.deque(
        maxlen=model.data_params.polls_per_window
    )
    analysis_thread = threading.Thread(
        target=workers.analysis_worker,
        args=(model, analysis_queue, kill_event),
        daemon=True
    )
    analysis_thread.start()

    # Parquet Writer Thread
    writer_queue = None
    writer_thread = None
    if config.write_to_file:
        file_name = f'{config.save_dir}/inputs_{time.strftime(formats.TIMESTAMP_FORMAT)}.parquet'
        schema = pyarrow.schema(
            fields=[
                (feature, pyarrow.float32()) 
                for feature in model.data_params.whitelist
            ],
            metadata={
                b'polling_rate': str(model.data_params.polling_rate).encode('utf-8')
            }
        )
        writer_queue = queue.Queue()
        writer_thread = threading.Thread(
            target=writers.parquet_writer_worker,
            args=(file_name, schema, writer_queue, kill_event),
            daemon=True
        )
        writer_thread.start()

    # Mouse Listener Thread
    mouse_listener_thread = None
    if any(bind in model.data_params.mouse_whitelist for bind in bind_enums.MouseAnalog):
        mouse_listener_thread = threading.Thread(
            target=windows_raw_mouse.listen_for_mouse_movement, 
            args=(kill_event,),
            daemon=True
        )
        mouse_listener_thread.start()

    poll_interval = 1.0 / model.data_params.polling_rate
    logger.info(f'Polling at {model.data_params.polling_rate}Hz (press {", ".join(config.kill_bind_list)} to stop)...')

    total_polls = 0
    try:
        while not poll_helpers.are_pressed(config.kill_bind_list, config.kill_bind_logic):
            time.sleep(poll_interval)
            poll = poll_helpers.poll_if_capturing(
                capture_binds=config.capture_binds, 
                capture_bind_gate=config.capture_bind_gate,
                data_params=model.data_params
            )
            if not poll:
                continue
            total_polls += 1
            
            if config.write_to_file:
                writer_queue.put(poll)
            
            window_ring_buffer.append(poll)
            if len(window_ring_buffer) < window_ring_buffer.maxlen:
                continue

            not_in_stride = (total_polls - model.data_params.polls_per_window) % model.data_params.window_stride != 0
            if not_in_stride:
                continue

            current_window = list(window_ring_buffer)
            if model.data_params.ignore_empty_polls:
                if not any(any(abs(val) > 1e-5 for val in poll) for poll in current_window):
                    continue
            
            analysis_queue.put(current_window)

        logger.info('Kill bind(s) detected. Stopping...')
    finally:
        kill_event.set()
        analysis_thread.join()
        if mouse_listener_thread:
            mouse_listener_thread.join()
        if config.write_to_file:
            writer_thread.join()
            logger.info(f'Data saved to {file_name}')