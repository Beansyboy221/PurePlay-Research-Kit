import sklearn.utils.validation
import sklearn.preprocessing
import sklearn.exceptions
import typing
import polars
import numpy

from source.utilities.file_utils.reading import (
    reader_registry,
    reader_outputs
)
from source.utilities.data_utils import dataparams

class ValidScaler(typing.Protocol):
    def fit(self) -> None: ...
    def partial_fit(self) -> None: ...
    def transform(self) -> numpy.ndarray: ...

class ScalerManager:
    def __init__(self, data_params: dataparams.DataParams):
        self.data_params = data_params
        self.scaler: ValidScaler = sklearn.preprocessing.RobustScaler()
        self._cache: list[ValidScaler] = []

    def fit(self, file_paths: list[str]) -> None:
        '''Fits the internal scaler to a specified list of files.'''
        for file_path in file_paths:
            reader = reader_registry.get_reader(file_path)
            if not isinstance(reader, reader_outputs.SupportsLazyFrame):
                raise ValueError(f'Reader for: {file_path} does not support lazy frames.')
            lazy_frame = reader.read_lazyframe(file_path, self.data_params.whitelist)
            if self.data_params.ignore_empty_polls:
                lazy_frame = lazy_frame.filter(
                    polars.sum_horizontal(self.data_params.whitelist) != 0
                )
            self.scaler.partial_fit(lazy_frame.collect().to_numpy().astype(numpy.float32))

    def load(self, scaler_class: type[ValidScaler], scaler_params: dict = None) -> None:
        '''Loads a scaler from cache if available, otherwise instantiates a new one.'''
        for cached_scaler in self._cache:
            if isinstance(cached_scaler, scaler_class):
                self.scaler = cached_scaler
                return
        self.scaler = scaler_class(**(scaler_params or {}))

    def fit_and_cache(self, file_paths: list[str]) -> None:
        '''Fits the current scaler and stores it in the cache.'''
        self.fit(file_paths)
        self._cache.append(self.scaler)

    def is_fitted(self) -> bool:
        '''Returns True if the current scaler has been fitted.'''
        try:
            sklearn.utils.validation.check_is_fitted(self.scaler)
            return True
        except sklearn.exceptions.NotFittedError:
            return False