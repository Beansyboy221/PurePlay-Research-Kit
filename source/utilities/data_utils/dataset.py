import sklearn
import polars
import torch
import numpy

from . import dataparams

class FileDataset(torch.utils.data.Dataset):
    '''Dataset for loading and processing a Parquet file.'''
    def __init__(
            self,
            file_path: str,
            data_params: dataparams.DataParams,
            scaler: sklearn.base.TransformerMixin,
            label: int = 0,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        metadata = polars.read_parquet_metadata(file_path)
        polling_rate = metadata.get('polling_rate')
        if not polling_rate:
            raise ValueError(f'Polling rate metadata is missing from file: {file_path}')
        reset_mouse_on_release = metadata.get('reset_mouse_on_release')
        if not reset_mouse_on_release:
            raise ValueError(f'Reset mouse on release metadata is missing from file: {file_path}')
        self.data_params = dataparams.ResolvedDataParams(
            **data_params.model_dump(),
            polling_rate=int(polling_rate),
            reset_mouse_on_release=reset_mouse_on_release.lower() == 'true'
        )

        data_frame = (
            polars.scan_parquet(file_path)
            .select(self.data_params.whitelist)
            .cast(polars.Float32)
            .collect()
        )
        data_array = scaler.transform(data_frame)

        self.data_tensor = self._create_windows(
            data=data_array, 
            window_size=data_params.polls_per_window,
            stride=data_params.window_stride,
            filter_empty=data_params.ignore_empty_polls
        )
        self.label_tensor = torch.tensor(label, dtype=torch.float32)

    def _create_windows(
            self,
            data: numpy.ndarray,
            window_size: int,
            stride: int,
            filter_empty: bool
        ) -> torch.Tensor:
        '''Creates a sliding window view of the data and filters out empty windows.'''
        if len(data) < window_size:
            raise ValueError(f'Insufficient data in {self.file_path}')
        
        windows = numpy.lib.stride_tricks.sliding_window_view(
            x=data, 
            window_shape=(window_size, data.shape[1])
        )
        windows = windows.squeeze(axis=1) # Remove extra dimension
        windows = windows[slice(None, None, stride)]

        if filter_empty:
            # 1. Identify which individual rows have any activity (O(N*Features))
            row_activity = numpy.any(numpy.abs(data) > 1e-5, axis=1)
            
            # 2. Check if ANY row within the window is active (O(N*WindowSize))
            window_masks = numpy.lib.stride_tricks.sliding_window_view(row_activity, window_size)
            window_masks = window_masks[slice(None, None, stride)]
            mask = numpy.any(window_masks, axis=1)
            
            windows = windows[mask]
        
        return torch.from_numpy(windows).clone()

    def __len__(self) -> int:
        return self.data_tensor.shape[0]

    def __getitem__(self, index: int):
        return self.data_tensor[index], self.label_tensor