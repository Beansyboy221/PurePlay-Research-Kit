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


SUPPORTED_SCALERS: dict[str, SupportedScaler] = {
    "standard": sklearn.preprocessing.StandardScaler(),
    "standard_sparse": sklearn.preprocessing.StandardScaler(with_mean=False),
    "min_max": sklearn.preprocessing.MinMaxScaler(),
    "robust": sklearn.preprocessing.RobustScaler(),
    "max_abs": sklearn.preprocessing.MaxAbsScaler(),
}
