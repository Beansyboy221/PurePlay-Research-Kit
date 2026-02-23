import threading
import pyarrow
import queue
import time
import utilities, devices, constants, logger

def collect_input_data(config: object) -> None:
    """Collects input data and saves it to a Parquet file."""
    file_name = f'{config.save_dir}/inputs_{time.strftime(constants.TIMESTAMP_FORMAT)}.parquet'
    whitelist = config.keyboard_whitelist + config.mouse_whitelist + config.gamepad_whitelist
    schema = pyarrow.schema(
        fields=[(feature, pyarrow.float32()) for feature in whitelist],
        metadata={b'polling_rate': str(config.polling_rate).encode('utf-8')}
    )

    data_queue = queue.Queue()
    kill_event = threading.Event()
    writer_thread = threading.Thread(
        target=utilities.parquet_writer_worker,
        args=(file_name, schema, data_queue, kill_event),
        daemon=True
    )
    writer_thread.start()
    
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
                data_queue.put(row)
            
            time.sleep(poll_interval)
    finally:
        kill_event.set()
        writer_thread.join()
        if mouse_listener_thread is not None:
            mouse_listener_thread.join()
        logger.info(f'Data saved to {file_name}')
