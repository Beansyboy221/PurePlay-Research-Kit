import pyarrow.parquet
import threading
import pyarrow
import typing
import queue
import time
import csv

#region Writers
class FileWriter(typing.Protocol):
    extensions: tuple[str, ...]

    def write_batch(self, batch: list) -> None: ...
    def close(self) -> None: ...

class ParquetWriter:
    extensions = ('.parquet',)

    def __init__(self, file_name: str, schema: pyarrow.Schema):
        self._writer = pyarrow.parquet.ParquetWriter(file_name, schema)
        self._schema = schema

    def write_batch(self, batch: list) -> None:
        table = pyarrow.Table.from_arrays(
            arrays=[
                pyarrow.array(column, type=pyarrow.float32())
                for column in zip(*batch)
            ],
            schema=self._schema
        )
        self._writer.write_table(table)

    def close(self) -> None:
        self._writer.close()

class CSVWriter:
    extensions = ('.csv',)

    def __init__(self, file_name: str, schema: pyarrow.Schema):
        self._file = open(file_name, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow([field.name for field in schema])  # header

    def write_batch(self, batch: list) -> None:
        self._writer.writerows(batch)

    def close(self) -> None:
        self._file.close()

class HDF5Writer:
    extensions = ('.h5', '.hdf5')

    def __init__(self, file_name: str, schema: pyarrow.Schema):
        import h5py
        self._file = h5py.File(file_name, 'w')
        self._schema = schema
        self._datasets = {
            field.name: self._file.create_dataset(
                field.name,
                shape=(0,),
                maxshape=(None,),
                dtype='float32'
            )
            for field in schema
        }

    def write_batch(self, batch: list) -> None:
        columns = list(zip(*batch))
        for field, column_data in zip(self._schema, columns):
            dataset = self._datasets[field.name]
            current_size = dataset.shape[0]
            new_size = current_size + len(column_data)
            dataset.resize((new_size,))
            dataset[current_size:new_size] = column_data

    def close(self) -> None:
        self._file.close()
#endregion

_WRITERS: list[FileWriter] = [ParquetWriter, CSVWriter, HDF5Writer]
SUPPORTED_EXTENSIONS: set[str] = {
    extension for writer in _WRITERS
    for extension in writer.extensions
}

def get_writer(file_name: str, schema: pyarrow.Schema) -> FileWriter:
    '''Automatically returns the right worker for the given file's extension.'''
    import os
    extension = os.path.splitext(file_name)[1].lower()
    for writer in _WRITERS:
        if extension in writer.extensions:
            return writer(file_name, schema)
    raise ValueError(f'No writer registered for extension: {extension}')

#region Worker
def file_writer_worker(
        file_name: str,
        schema: pyarrow.Schema,
        data_queue: queue.Queue,
        kill_event: threading.Event
    ) -> None:
    '''Worker function to write polled data to a file of any supported format.'''
    writer = get_writer(file_name, schema)
    try:
        while not kill_event.is_set():
            batch = []
            while not data_queue.empty():
                try:
                    batch.append(data_queue.get_nowait())
                except queue.Empty:
                    break
            if batch:
                writer.write_batch(batch)
            if not kill_event.is_set():
                time.sleep(1.0)
    finally:
        writer.close()
#endregion