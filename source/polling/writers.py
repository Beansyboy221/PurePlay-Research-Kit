import pyarrow.parquet
import threading
import pyarrow
import queue
import time

def get_queued_batch(data_queue: queue.Queue, batch_size: int):
    '''Helper function to get a batch of data from the queue.'''
    batch = []
    try:
        while len(batch) < batch_size:
            batch.append(data_queue.get_nowait())
    except queue.Empty:
        pass
    return batch

def parquet_writer_worker(
        file_name: str, 
        schema: pyarrow.schema, 
        data_queue: queue.Queue, 
        kill_event: threading.Event, 
        write_delay: float = 1.0, 
        batch_size: int = 100
    ) -> None:
    '''Worker function to write batched data to a file.'''
    writer = pyarrow.parquet.ParquetWriter(file_name, schema)
    with writer:
        while not kill_event.is_set():
            time.sleep(write_delay)

            batch = get_queued_batch(data_queue, batch_size)
            if not batch:
                continue

            table = pyarrow.Table.from_arrays(
                arrays=[pyarrow.array(column, type=pyarrow.float32()) for column in zip(*batch)], 
                schema=schema
            )
            writer.write_table(table)