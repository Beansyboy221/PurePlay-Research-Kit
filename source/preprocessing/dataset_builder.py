"""A builder class for datasets."""

import sklearn.utils.validation
import pydantic
import os

from . import datamodule, dataset


class DatasetBuilder:
    def __init__(self, data_module: datamodule.PurePlayDataModule):
        self.data_module = data_module

    def get_files_from_dir(
        self, directory: pydantic.DirectoryPath
    ) -> list[pydantic.FilePath]:
        """Returns a list of paths for all files in the given directory."""
        return [
            pydantic.FilePath(os.path.join(directory, file_path))
            for file_path in os.listdir(directory)
            if file_path.endswith(".parquet")
        ]

    def make_datasets(
        self, labeled_dirs: dict[pydantic.DirectoryPath, int]
    ) -> list[dataset.ParquetDataset]:
        """Creates datasets for a map of given labeled directories."""
        sklearn.utils.validation.check_is_fitted(
            estimator=self.data_module.scaler,
            msg="Scaler must be fitted before creating datasets.",
        )
        return [
            dataset.ParquetDataset(
                file_path=file_path,
                data_params=self.data_module.params,
                scaler=self.data_module.scaler,
                label=label,
            )
            for directory, label in labeled_dirs.items()
            for file_path in self.get_files_from_dir(directory)
        ]

    def check_consistency(self, datasets: list[dataset.ParquetDataset]) -> None:
        """Checks that all dataset params match."""
        if not datasets:
            return

        file_fields = {"polling_rate", "reset_mouse_on_release"}
        for dataset in datasets:
            if dataset.params.model_dump(
                exclude=file_fields
            ) != self.data_module.params.model_dump(exclude=file_fields):
                raise ValueError(f"Mismatched params in: {dataset.file_path}")
