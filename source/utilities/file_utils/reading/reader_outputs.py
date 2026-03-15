import typing
import polars

@typing.runtime_checkable
class SupportsMetadata(typing.Protocol):
    def read_metadata(self, file_path: str) -> dict: ...

@typing.runtime_checkable
class SupportsDict(typing.Protocol):
    def read_dict(self, file_path: str) -> dict[str, typing.Any]: ...

@typing.runtime_checkable
class SupportsLazyFrame(typing.Protocol):
    def read_lazyframe(self, file_path: str, columns: list[str]) -> polars.LazyFrame: ...

@typing.runtime_checkable
class SupportsDataFrame(typing.Protocol):
    def read_dataframe(self, file_path: str, columns: list[str]) -> polars.DataFrame: ...