import pyarrow.parquet
import threading
import pyarrow
import queue
import time
    
def parquet_writer_worker(
        file_name: str, 
        schema: pyarrow.schema, 
        data_queue: queue, 
        kill_event: threading.Event
    ) -> None:
    """Worker function to write polled data to a Parquet file."""
    with pyarrow.parquet.ParquetWriter(file_name, schema) as writer:
        while not kill_event.is_set():
            batch = []
            while not data_queue.empty():
                try:
                    batch.append(data_queue.get_nowait())
                except queue.Empty:
                    break
            
            if batch:
                table = pyarrow.Table.from_arrays(
                    arrays=[pyarrow.array(column, type=pyarrow.float32()) for column in zip(*batch)], 
                    schema=schema
                )
                writer.write_table(table)
            
            if not kill_event.is_set():
                time.sleep(1.0) # Should this be relative to config.polling_rate?