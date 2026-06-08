import threading
import logging
import queue
import time

import pyarrow

from beaninput import helpers as poll_helpers

from misc import writers, formats

from . import config


def main(config: config.CollectConfig) -> None:
    """
    Main entry point for collect mode.
    Collects input data and saves it to a Parquet file.
    """
    logger = logging.getLogger()

    logger.info("Starting writer thread...")
    kill_event = threading.Event()
    save_path = (
        f"{config.save_dir}/inputs_{time.strftime(formats.TIMESTAMP_FORMAT)}.parquet"
    )
    schema = pyarrow.schema(
        fields=[(feature, pyarrow.float32()) for feature in config.whitelist],
        metadata={b"polling_rate": str(config.polling_rate).encode("utf-8")},
    )
    writer_queue = queue.Queue()
    writer_thread = threading.Thread(
        target=writers.parquet_writer_worker,
        args=(save_path, schema, writer_queue, kill_event),
        daemon=True,
    )
    writer_thread.start()

    logger.info("Starting device listeners...")
    poll_helpers.start_listeners()

    poll_interval = 1.0 / config.polling_rate
    logger.info(
        f'Polling at {config.polling_rate}Hz (press {", ".join(config.kill_binds)} to stop)...'
    )
    try:
        while not poll_helpers.are_active(config.kill_binds, config.kill_bind_logic):
            if poll := poll_helpers.poll_if_capturing(
                capture_binds=config.capture_binds,
                capture_bind_gate=config.capture_bind_gate,
                data_params=config,
            ):
                writer_queue.put(poll)
            time.sleep(poll_interval)
    finally:
        logger.info("Kill bind(s) detected. Stopping...")
        kill_event.set()
        writer_thread.join()
        poll_helpers.stop_listeners()
        logger.info(f"Data saved to file: {save_path}")
