import collections
import threading
import pyarrow
import logging
import queue
import time

from globals import formats
from misc import logging_utils, cuda_helpers, writers
import models
from polling import helpers
from polling.mouse import helpers as mouse_helpers, binds
from . import config, workers


def main(config: config.DeployConfig) -> None:
    """
    Main entry point for deployment mode.
    Performs live analysis on input data using a pre-trained model.
    """
    logger = logging_utils.get_logger()
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    logger.info("Configuring CUDA for your hardware...")
    cuda_helpers.optimize_cuda_for_hardware()

    logger.info(f"Loading model from file: {config.model_file}")
    model = models.load_model(config.model_file)
    config.ignore_empty_polls = model.data_params.ignore_empty_polls
    kill_event = threading.Event()

    logger.info("Starting analysis thread...")
    analysis_queue = queue.Queue()
    window_ring_buffer = collections.deque(maxlen=model.data_params.polls_per_window)
    analysis_thread = threading.Thread(
        target=workers.analysis_worker,
        args=(model, analysis_queue, kill_event),
        daemon=True,
    )
    analysis_thread.start()

    writer_queue = None
    writer_thread = None
    if config.write_to_file:
        logger.info("Starting writer thread...")
        file_name = f"{config.save_dir}/inputs_{time.strftime(formats.TIMESTAMP_FORMAT)}.parquet"
        schema = pyarrow.schema(
            fields=[
                (feature, pyarrow.float32()) for feature in model.data_params.whitelist
            ],
            metadata={
                b"polling_rate": str(model.data_params.polling_rate).encode("utf-8")
            },
        )
        writer_queue = queue.Queue()
        writer_thread = threading.Thread(
            target=writers.parquet_writer_worker,
            args=(file_name, schema, writer_queue, kill_event),
            daemon=True,
        )
        writer_thread.start()

    if any(bind in model.data_params.mouse_whitelist for bind in binds.Button.registry):
        logger.info("Starting mouse button listener...")
        mouse_helpers.mouse_button_listener.start()
    if any(bind in model.data_params.mouse_whitelist for bind in binds.Move.registry):
        logger.info("Starting mouse movement listener...")
        mouse_helpers.mouse_move_listener.start()

    poll_interval = 1.0 / model.data_params.polling_rate
    logger.info(
        f'Polling at {model.data_params.polling_rate}Hz (press {", ".join(config.kill_binds)} to stop)...'
    )
    total_polls = 0
    try:
        while not helpers.are_active(config.kill_binds, config.kill_bind_logic):
            time.sleep(poll_interval)
            poll = helpers.poll_if_capturing(
                capture_binds=config.capture_binds,
                capture_bind_gate=config.capture_bind_gate,
                data_params=model.data_params,
            )
            if not poll:
                continue
            total_polls += 1

            if config.write_to_file:
                writer_queue.put(poll)

            window_ring_buffer.append(poll)
            if len(window_ring_buffer) < window_ring_buffer.maxlen:
                continue

            not_in_stride = (
                total_polls - model.data_params.polls_per_window
            ) % model.data_params.window_stride != 0
            if not_in_stride:
                continue

            current_window = list(window_ring_buffer)
            if model.data_params.ignore_empty_polls:
                if not any(
                    any(abs(val) > 1e-5 for val in poll) for poll in current_window
                ):
                    continue

            analysis_queue.put(current_window)
    finally:
        logger.info("Kill bind(s) detected. Stopping...")
        kill_event.set()
        analysis_thread.join()
        mouse_helpers.mouse_button_listener.stop()
        mouse_helpers.mouse_move_listener.stop()
        if config.write_to_file:
            writer_thread.join()
            logger.info(f"Data saved to {file_name}")
