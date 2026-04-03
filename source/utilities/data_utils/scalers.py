import sklearn.preprocessing

SUPPORTED_SCALERS: frozenset[type[sklearn.base.TransformerMixin]] = (
    sklearn.preprocessing.StandardScaler, # May want to consider having another version where with_mean=False.
    sklearn.preprocessing.MinMaxScaler,
    sklearn.preprocessing.MaxAbsScaler # I think this will be the best choice.
)