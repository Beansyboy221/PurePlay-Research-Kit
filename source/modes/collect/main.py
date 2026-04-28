import threading
import pyarrow
import queue
import time

from globals import formats
from misc import logging_utils
from polling import (
    unified_helpers,
    writers
)
from polling.mouse import (
    helpers as mouse_helpers,
    binds
)
from . import config

def main(config: config.CollectConfig) -> None:
    '''
    Main entry point for collect mode.
    Collects input data and saves it to a Parquet file.
    '''
    logger = logging_utils.get_logger()

    # Should these be sorted here or in the config?
    sorted_keyboard_whitelist = sorted(config.keyboard_whitelist, key=lambda x: x.name) # Maybe make whitelists into dicts?
    sorted_mouse_whitelist = sorted(config.mouse_whitelist, key=lambda x: x.name)
    sorted_controller_whitelist = sorted(config.controller_whitelist, key=lambda x: x.name)

    kill_event = threading.Event()
    file_name = f'{config.save_dir}/inputs_{time.strftime(formats.TIMESTAMP_FORMAT)}.parquet'
    whitelist = sorted_keyboard_whitelist + sorted_mouse_whitelist + sorted_controller_whitelist
    schema = pyarrow.schema(
        fields=[(feature, pyarrow.float32()) for feature in whitelist],
        metadata={b'polling_rate': str(config.polling_rate).encode('utf-8')}
    )
    data_queue = queue.Queue()
    writer_thread = threading.Thread(
        target=writers.parquet_writer_worker,
        args=(file_name, schema, data_queue, kill_event),
        daemon=True
    )
    logger.info('Starting writer thread...')
    writer_thread.start()
    
    if any(bind in sorted_mouse_whitelist for bind in binds.MouseButton):
        logger.info('Starting mouse button listener...')
        mouse_helpers.mouse_button_listener.start()
    if any(bind in sorted_mouse_whitelist for bind in binds.MouseMove):
        logger.info('Starting mouse movement listener...')
        mouse_helpers.mouse_move_listener.start()
    
    poll_interval = 1.0 / config.polling_rate
    logger.info(f'Polling at {config.polling_rate}Hz (press {", ".join(config.kill_bind_list)} to stop)...')
    try:
        while not unified_helpers.are_active(config.kill_bind_list, config.kill_bind_logic):
            poll = unified_helpers.poll_if_capturing(
                capture_binds=config.capture_binds, 
                capture_bind_gate=config.capture_bind_gate,
                data_params=config
            )
            if poll:
                data_queue.put(poll)
            time.sleep(poll_interval)
    finally:
        logger.info('Kill bind(s) detected. Stopping...')
        kill_event.set()
        writer_thread.join()
        mouse_helpers.mouse_button_listener.stop()
        mouse_helpers.mouse_move_listener.stop()
        logger.info(f'Data saved to {file_name}')