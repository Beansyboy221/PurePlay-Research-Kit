import polars
import torch
import numpy

# PurePlay imports
from source.utilities.file_utils.reading import reader_registry
from source.utilities.data_utils import (
    dataparams
)   

class FileDataset(torch.utils.data.Dataset):
    '''Dataset for loading and processing a Parquet file.'''
    def __init__(
            self,
            file_path: str,
            data_params: dataparams.DataParams,
            scaler: object,
            label: int = 0
        ):
        # Read dataset-defined params from file metadata
        self.reader = reader_registry.get_reader(file_path)
        metadata = self.reader.read_metadata(file_path)
        polling_rate = metadata.get(b'polling_rate')
        if polling_rate is None:
            raise ValueError(f'Polling rate metadata is missing from file: {file_path}')
        reset_mouse_on_release = metadata.get(b'reset_mouse_on_release')
        if reset_mouse_on_release is None:
            raise ValueError(f'Reset mouse on release metadata is missing from file: {file_path}')

        # Resolve data params
        self.data_params = dataparams.ResolvedDataParams(
            **data_params.model_dump(),
            polling_rate=int(polling_rate.decode('utf-8')),
            reset_mouse_on_release=reset_mouse_on_release.decode('utf-8').lower() == 'true'
        )
        
        # Load file data
        lazy_frame = self.reader.read_lazy(file_path, data_params.whitelist)

        # Filter out empty polls if needed 
        # Should I filter out whole sequences to preserve temporal structure?                            IMPORTANT
        if data_params.ignore_empty_polls:
            lazy_frame = lazy_frame.filter(
                polars.sum_horizontal(data_params.whitelist) != 0
            )

        # Convert to a numpy array
        data_array = lazy_frame.collect().to_numpy(writable=True).astype(numpy.float32)
        
        # Scale the data
        data_array = scaler.transform(data_array)
        
        # Trim excess rows to make full sequences
        num_sequences = len(data_array) // data_params.polls_per_sequence
        total_rows = num_sequences * data_params.polls_per_sequence
        data_array = data_array[:total_rows].reshape(num_sequences, data_params.polls_per_sequence, -1)

        # Convert to PyTorch tensors
        self.data_tensor = torch.from_numpy(data_array)
        self.label_tensor = torch.tensor(label, dtype=torch.float32)

        # Set metadata
        self.length = len(self.data_tensor)
        self.file_path = file_path
        self.label = label

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor