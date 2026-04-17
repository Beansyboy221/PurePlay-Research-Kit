import threading
import pyarrow
import queue
import time

from globals import (
    formats,
    logger
)
from data import writers
from data.polling import (
    windows_raw_mouse,
    poll_helpers,
    bind_enums
)
from . import collect_config

def collect(config: collect_config.CollectConfig) -> None:
    '''
    Main entry point for collect mode.
    Collects input data and saves it to a Parquet file.
    '''
    # Should these be sorted here or in the config? I think in the config is better.
    sorted_keyboard_whitelist = sorted(config.keyboard_whitelist, key=lambda x: x.name)
    sorted_mouse_whitelist = sorted(config.mouse_whitelist, key=lambda x: x.name)
    sorted_gamepad_whitelist = sorted(config.gamepad_whitelist, key=lambda x: x.name)

    kill_event = threading.Event()
    file_name = f'{config.save_dir}/inputs_{time.strftime(formats.TIMESTAMP_FORMAT)}.parquet'
    whitelist = sorted_keyboard_whitelist + sorted_mouse_whitelist + sorted_gamepad_whitelist
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
    writer_thread.start()
    
    mouse_listener_thread = None
    if any(bind in sorted_mouse_whitelist for bind in bind_enums.MouseAnalog):
        mouse_listener_thread = threading.Thread(
            target=windows_raw_mouse.listen_for_mouse_movement, 
            args=(kill_event,),
            daemon=True
        )
        mouse_listener_thread.start()
    
    poll_interval = 1.0 / config.polling_rate
    logger.info(f'Polling at {config.polling_rate}Hz (press {", ".join(config.kill_bind_list)} to stop)...')

    try:
        while not poll_helpers.are_pressed(config.kill_bind_list, config.kill_bind_logic):
            poll = poll_helpers.poll_if_capturing(
                capture_binds=config.capture_binds, 
                capture_bind_gate=config.capture_bind_gate,
                data_params=config
            )
            if poll:
                data_queue.put(poll)
            time.sleep(poll_interval)
        logger.info('Kill bind(s) detected. Stopping...')
    finally:
        kill_event.set()
        writer_thread.join()
        if mouse_listener_thread:
            mouse_listener_thread.join()
        logger.info(f'Data saved to {file_name}')