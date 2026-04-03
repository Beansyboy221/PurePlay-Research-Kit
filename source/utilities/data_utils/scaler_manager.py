import sklearn.preprocessing
import pydantic
import polars

from . import data_params

class ScalerManager:
    def __init__(self, data_params: data_params.DataParams):
        self.data_params = data_params
        self.scaler: sklearn.base.TransformerMixin = sklearn.preprocessing.StandardScaler()
        self._cache: list[sklearn.base.TransformerMixin] = []

    def fit(self, file_paths: list[pydantic.FilePath]) -> None:
        '''Fits the internal scaler to a specified list of files.'''
        for file_path in file_paths:
            data_frame = (
                polars.scan_parquet(file_path)
                .select(self.data_params.whitelist)
                .cast(polars.Float32)
                .collect()
            )
            self.scaler.partial_fit(data_frame)
    
    def fit_and_cache(self, file_paths: list[pydantic.FilePath]) -> None:
        '''Fits the current scaler and stores it in the cache.'''
        self.fit(file_paths)
        self._cache.append(self.scaler)

    def load(self, scaler_class: type[sklearn.base.TransformerMixin]) -> None:
        '''Loads a scaler from cache if available, otherwise instantiates a new one.'''
        for cached_scaler in self._cache:
            if isinstance(cached_scaler, scaler_class):
                self.scaler = cached_scaler
                return
        self.scaler = scaler_class()