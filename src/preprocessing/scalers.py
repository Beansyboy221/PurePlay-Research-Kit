"""Scalers used by PurePlay."""

import typing
import numpy

import sklearn.preprocessing
import pandas


@typing.runtime_checkable
class SupportedScaler(typing.Protocol):
    def fit(self, X) -> typing.Self: ...
    def partial_fit(self, X) -> typing.Self: ...
    def transform(self, X) -> typing.Union[numpy.ndarray, pandas.DataFrame]: ...


SCALER_CACHE: frozenset[SupportedScaler] = (
    sklearn.preprocessing.StandardScaler(),
    sklearn.preprocessing.StandardScaler(with_mean=False),
    sklearn.preprocessing.MinMaxScaler(),
    sklearn.preprocessing.RobustScaler(),
    sklearn.preprocessing.MaxAbsScaler(),
)
