'''
Usage:
Call get_reader and check if it supports your output type (with protocols and isinstance). 
Alternatively, manually instantiate a reader for your file(s).
You can also get a list of supported extensions for a given output type.

Example:
reader = file_readers.get_reader(file_path)
if not isinstance(reader, file_readers.SupportsLazyFrame):
    raise ValueError(f'Reader for: {file_path} does not support lazy frames.')
lazy_frame = reader.read_lazyframe(file_path, self.data_params.whitelist)
'''

import typing
import polars

from reading import (
    reader_registry,
    base_reader
)

@reader_registry.register()
class ParquetReader(base_reader.FileReader):
    extensions = ('.parquet',)
    
    def read_metadata(self, file_path: str) -> dict:
        import pyarrow.parquet
        raw = pyarrow.parquet.read_metadata(file_path).metadata
        return {
            key.decode(): value.decode() 
            for key, value in raw.items()
        }

    def read_lazyframe(self, file_path: str, columns: list[str]) -> polars.LazyFrame:
        return polars.scan_parquet(file_path).select(columns)
    
    def read_dataframe(self, file_path: str, columns: list[str]) -> polars.DataFrame:
        return polars.read_parquet(file_path).select(columns)

@reader_registry.register()
class CSVReader(base_reader.FileReader):
    extensions = ('.csv',)

    def read_metadata(self, file_path: str) -> dict:
        import json
        # CSVs have no native metadata — use a sidecar .json file by convention
        meta_path = file_path.replace('.csv', '.meta.json')
        with open(meta_path) as file_handle:
            return json.load(file_handle)

    def read_lazyframe(self, file_path: str, columns: list[str]) -> polars.LazyFrame:
        return polars.scan_csv(file_path).select(columns)
    
    def read_dataframe(self, file_path: str, columns: list[str]) -> polars.DataFrame:
        return polars.read_csv(file_path).select(columns)

@reader_registry.register()
class HDF5Reader(base_reader.FileReader):
    extensions = ('.h5', '.hdf5')

    def read_metadata(self, file_path: str) -> dict:
        import h5py
        with h5py.File(file_path, 'r') as file_handle:
            return dict(file_handle.attrs)

    def read_lazyframe(self, file_path: str, columns: list[str]) -> polars.LazyFrame:
        import h5py
        with h5py.File(file_path, 'r') as file_handle:
            data = {
                column: file_handle[column][:]
                for column in columns
            }
        return polars.LazyFrame(data)
    
    def read_dataframe(self, file_path: str, columns: list[str]) -> polars.DataFrame:
        import h5py
        with h5py.File(file_path, 'r') as file_handle:
            data = {
                column: file_handle[column][:]
                for column in columns
            }
        return polars.DataFrame(data)

@reader_registry.register()
class JSONReader(base_reader.FileReader):
    extensions = ('.json',)

    def read_dict(self, file_path: str) -> dict[str, typing.Any]:
        import json
        with open(file_path, 'r', encoding='utf-8') as file_handle:
            return json.load(file_handle)

@reader_registry.register()
class TOMLReader(base_reader.FileReader):
    extensions = ('.toml',)

    def read_dict(self, file_path: str) -> dict[str, typing.Any]:
        import tomllib
        with open(file_path, 'rb') as file_handle:
            return tomllib.load(file_handle)

@reader_registry.register()
class YAMLReader(base_reader.FileReader):
    extensions = ('.yaml', '.yml')

    def read_dict(self, file_path: str) -> dict[str, typing.Any]:
        import yaml
        with open(file_path, 'r', encoding='utf-8') as file_handle:
            return yaml.safe_load(file_handle)

@reader_registry.register()
class INIReader(base_reader.FileReader):
    extensions = ('.ini', '.conf')

    def read_dict(self, file_path: str) -> dict[str, typing.Any]:
        import configparser
        config = configparser.ConfigParser()
        config.read(file_path)
        return {
            section: dict(config[section]) 
            for section in config.sections()
        }