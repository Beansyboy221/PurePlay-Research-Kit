"""A torch.utils.data.Dataset subclass that loads and preprocesses a single PurePlay data file."""

import polars
import torch
import numpy

from misc import logging_utils
from polling import base_bind
from . import scalers, params

logger = logging_utils.get_logger()


class ParquetDataset(torch.utils.data.Dataset):
    """Dataset for loading and preprocessing a Parquet file."""

    @staticmethod
    def load_file(file_path: str, whitelist: list[base_bind.Bind]) -> polars.DataFrame:
        """Centralized method to load and cast data consistently."""
        return (
            polars.scan_parquet(file_path)
            .select(whitelist)
            .cast(polars.Float32)
            .collect()
        )

    def __init__(
        self,
        file_path: str,
        chosen_params: params.ProcessingParams,
        scaler: scalers.SupportedScaler,
        label: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        logger.info(f"Creating dataset from file: {file_path}...")

        logger.info("Loading metadata from file...")
        metadata = polars.read_parquet_metadata(file_path)
        if not metadata:
            raise RuntimeError("File is missing metadata.")
        self.params = params.DataParams(
            **chosen_params.model_dump(),
            polling_rate=int(metadata.get("polling_rate")),
            reset_mouse_on_release=metadata.get("reset_mouse_on_release").lower()
            == "true",
        )

        logger.info("Loading data from file...")
        data_frame = self.load_file(file_path, self.params.whitelist)

        logger.info("Scaling data...")
        data_array = scaler.transform(data_frame)

        logger.info("Breaking the data into windows...")
        if len(data_array) < chosen_params.polls_per_window:
            raise ValueError(f"Insufficient data in {self.file_path}")
        windows = numpy.lib.stride_tricks.sliding_window_view(
            x=data_array,
            window_shape=(chosen_params.polls_per_window, data_array.shape[1]),
        )
        windows = windows.squeeze(1)  # Remove extra dimension
        windows = windows[:: chosen_params.window_stride]
        if chosen_params.ignore_empty_polls:
            # 1. Identify which individual rows have any activity (O(N*Features))
            row_activity = numpy.any(numpy.abs(data_array) > 1e-5, axis=1)

            # 2. Check if ANY row within the window is active (O(N*WindowSize))
            window_masks = numpy.lib.stride_tricks.sliding_window_view(
                row_activity, chosen_params.polls_per_window
            )
            window_masks = window_masks[:: chosen_params.window_stride]
            mask = numpy.any(window_masks, axis=1)

            windows = windows[mask]
        self.data_tensor = torch.from_numpy(windows).clone()
        self.label_tensor = torch.tensor(label, dtype=torch.float32)

        logger.info("Dataset initialized.")

    def __len__(self) -> int:
        return self.data_tensor.shape[0]

    def __getitem__(self, index: int):
        return self.data_tensor[index], self.label_tensor
