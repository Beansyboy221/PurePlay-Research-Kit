import collections
import threading
import pyarrow
import logging
import queue
import time

from globals.constants import formats
from globals.enums import binds
from source.utilities.app_utils import cuda_helpers
from utilities.app_utils import (
    config_helpers,
    global_logger
)
from utilities.data_utils import (
    windows_raw_mouse,
    async_writers,
    poll_helpers
)
from source.ml_models import model_utils
from . import (
    workers,
    config,
    enums
)

def deploy(config: config.ModeConfig) -> None:
    '''
    Main entry point for deployment mode.
    Performs live analysis on input data using a pre-trained model.
    '''
    cuda_helpers.optimize_cuda_for_hardware()
    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    model = model_utils.load_model(config.model_file)
    kill_event = threading.Event()
    
    # Analysis Thread
    data_queue = queue.Queue()
    window_buffer = collections.deque(
        maxlen=2*model.data_params.polls_per_window
    )
    analysis_thread = threading.Thread(
        target=workers.analysis_worker,
        args=(model, data_queue, kill_event),
        daemon=True
    )
    analysis_thread.start()

    # Parquet Writer Thread
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
        data_queue = queue.Queue()
        writer_thread = threading.Thread(
            target=async_writers.parquet_writer_worker,
            args=(file_name, schema, data_queue, kill_event),
            daemon=True
        )
        writer_thread.start()

    # Mouse Listener Thread
    mouse_listener_thread = None
    if any(bind in model.data_params.mouse_whitelist for bind in binds.MouseAnalog):
        mouse_listener_thread = threading.Thread(
            target=windows_raw_mouse.listen_for_mouse_movement, 
            args=(kill_event,),
            daemon=True
        )
        mouse_listener_thread.start()

    poll_interval = 1.0 / model.data_params.polling_rate
    global_logger.info(f'Polling at {model.data_params.polling_rate}Hz (press {", ".join(config.kill_bind_list)} to stop)...')

    try:
        while not poll_helpers.should_kill(
            kill_binds=config.kill_bind_list, 
            kill_bind_gate=config.kill_bind_logic
        ):
            time.sleep(poll_interval)
            row = poll_helpers.poll_if_capturing(
                config.capture_bind_list, 
                config.capture_bind_logic, 
                model.data_params.keyboard_whitelist,
                model.data_params.mouse_whitelist,
                model.data_params.gamepad_whitelist,
                model.data_params.reset_mouse_on_release
            )
            if not row:
                continue

            if config.write_to_file:
                data_queue.put(row)
            window_buffer.append(row)
            if len(window_buffer) < model.data_params.polls_per_window:
                continue

            match config.deployment_window_type:
                case enums.WindowType.TUMBLING:
                    data_queue.put(list(window_buffer)[:model.data_params.polls_per_window])
                    window_buffer.clear()
                case enums.WindowType.SLIDING:
                    data_queue.put(list(window_buffer)[-model.data_params.polls_per_window:])
        global_logger.info('Kill bind(s) detected. Stopping...')
    finally:
        kill_event.set()
        analysis_thread.join()
        if mouse_listener_thread:
            mouse_listener_thread.join()
        if config.write_to_file:
            writer_thread.join()
            global_logger.info(f'Data saved to {file_name}')

if __name__ == '__main__':
    try:
        args = config_helpers.parse_args()
        config_path, use_gui, log_level = config_helpers.get_global_configs(args)
        global_logger.set_log_level(log_level)
        config_dict = config_helpers.load_config_file(config_path)
        config_dict = config_helpers.populate_missing_fields(config.ModeConfig, config_dict, use_gui)
        config_object = config.ModeConfig.model_validate(config_dict)
        deploy(config_object)
    except Exception as e:
        global_logger.exception(e)