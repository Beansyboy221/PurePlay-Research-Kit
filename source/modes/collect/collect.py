import threading
import pyarrow
import queue
import time

# PurePlay imports
from globals.constants import formats
from globals.enums import binds
from source.file_utils.writing import file_writers
from source.globals import global_logger
from utilities import (
    windows_raw_mouse,
    config_utils,
    config_utils, 
    poll_utils
)

# Mode-specific imports
from .config import ModeConfig

def collect(config: ModeConfig) -> None:
    '''
    Main entry point for collect mode.
    Collects input data and saves it to a Parquet file.
    '''
    kill_event = threading.Event()
    file_name = f'{config.save_dir}/inputs_{time.strftime(formats.TIMESTAMP_FORMAT)}.parquet'
    whitelist = config.keyboard_whitelist + config.mouse_whitelist + config.gamepad_whitelist
    schema = pyarrow.schema(
        fields=[(feature, pyarrow.float32()) for feature in whitelist],
        metadata={b'polling_rate': str(config.polling_rate).encode('utf-8')}
    )
    data_queue = queue.Queue()
    writer_thread = threading.Thread(
        target=file_writers.file_writer_worker,
        args=(file_name, schema, data_queue, kill_event),
        daemon=True
    )
    writer_thread.start()
    
    mouse_listener_thread = None
    if any(bind in config.mouse_whitelist for bind in binds.MouseAnalog):
        mouse_listener_thread = threading.Thread(
            target=windows_raw_mouse.listen_for_mouse_movement, 
            args=(kill_event,),
            daemon=True
        )
        mouse_listener_thread.start()
    
    poll_interval = 1.0 / config.polling_rate
    global_logger.info(f'Polling at {config.polling_rate}Hz (press {", ".join(config.kill_bind_list)} to stop)...')

    try:
        while True:
            if poll_utils.should_kill(
                    kill_binds=config.kill_bind_list, 
                    kill_bind_gate=config.kill_bind_logic
                ):
                global_logger.info('Kill bind(s) detected. Stopping...')
                break
            
            row = poll_utils.poll_if_capturing(
                config.capture_bind_list, 
                config.capture_bind_logic, 
                config.keyboard_whitelist, 
                config.mouse_whitelist, 
                config.gamepad_whitelist,
                config.reset_mouse_on_release
            )
            if row:
                data_queue.put(row)
            
            time.sleep(poll_interval)
    finally:
        kill_event.set()
        writer_thread.join()
        if mouse_listener_thread is not None:
            mouse_listener_thread.join()
        global_logger.info(f'Data saved to {file_name}')

if __name__ == '__main__':
    try:
        args = config_utils.parse_args()
        config_path, use_gui, log_level = config_utils.get_global_configs(args)
        global_logger.set_log_level(log_level)
        config_dict = config_utils.load_config_file(config_path)
        config_dict = config_utils.populate_missing_fields(ModeConfig, config_dict, use_gui)
        config_object = ModeConfig.model_validate(config_dict)
        collect(config_object)
    except Exception as e:
        global_logger.exception(e)