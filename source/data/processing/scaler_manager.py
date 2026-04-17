import polars

from misc import validated_paths
from . import (
    data_params,
    scalers
)

class ScalerManager:
    def __init__(self, data_params: data_params.DataParams):
        self.data_params = data_params
        self.scaler: scalers.SupportedScaler = scalers.SCALER_CACHE[0]

    def fit(self, file_paths: list[validated_paths.ParquetPath]) -> None:
        '''Fits the internal scaler to a specified list of files.'''
        for file_path in file_paths:
            data_frame = (
                polars.scan_parquet(file_path)
                .select(self.data_params.whitelist)
                .cast(polars.Float32)
                .collect()
            )
            self.scaler.partial_fit(data_frame)

    def load(self, scaler_class: type[scalers.SupportedScaler]) -> None:
        '''Loads a scaler from cache if available, otherwise instantiates a new one.'''
        for cached_scaler in scalers.SCALER_CACHE:
            if isinstance(cached_scaler, scaler_class):
                self.scaler = cached_scaler
                return
        self.scaler = scaler_class()