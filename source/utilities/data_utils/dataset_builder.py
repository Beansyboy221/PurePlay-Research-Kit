import sklearn.utils.validation
import polars
import os

from source.utilities.file_utils.reading import base_reader
from source.utilities.data_utils import (
    scaler_manager,
    dataparams, 
    datasets
)

class DatasetBuilder:
    def __init__(
            self,
            data_params: dataparams.DataParams,
            scaler_manager: scaler_manager.ScalerManager
        ):
        self.data_params = data_params
        self.scaler_manager = scaler_manager

    def get_files_from_dir(self, directory: str) -> list[str]:
        '''Returns a list of file paths for all supported files in the given directory.'''
        return [
            os.path.join(directory, file_path)
            for file_path in os.listdir(directory)
            if os.path.splitext(file_path)[1].lower() in base_reader.get_supported_extensions(polars.LazyFrame)
        ]

    def make_datasets(self, file_paths: list[str], label: int) -> list[datasets.FileDataset]:
        '''Creates FileDataset instances for each given file path.'''
        sklearn.utils.validation.check_is_fitted(
            estimator=self.scaler_manager.scaler,
            msg='Scaler must be fitted before creating datasets.'
        )
        return [
            datasets.FileDataset(
                file_path=file_path,
                data_params=self.data_params,
                scaler=self.scaler_manager.scaler,
                label=label
            )
            for file_path in file_paths
        ]

    def make_datasets_from_labeled_dirs(
            self,
            labeled_dirs: dict[str, int]
        ) -> list[datasets.FileDataset]:
        '''Creates datasets for all directories, assigning each its corresponding label.'''
        result = []
        for directory, label in labeled_dirs.items():
            result += self.make_datasets(self.get_files_from_dir(directory), label=label)
        return result

    def check_consistency(self, datasets_list: list[datasets.FileDataset]) -> None:
        '''Checks that all dataset params match.'''
        if not datasets_list:
            return

        def check_fields(fields, reference_params, check_datasets):
            for field in fields:
                reference_value = getattr(reference_params, field)
                for dataset in check_datasets:
                    if getattr(dataset.data_params, field) != reference_value:
                        raise ValueError(
                            f'Mismatched property: {field} found in: {dataset.file_path}'
                        )

        check_fields(dataparams.DataParams.model_fields, self.data_params, datasets_list)

        resolved_only_fields = (
            dataparams.ResolvedDataParams.model_fields.keys() 
            - dataparams.DataParams.model_fields.keys()
        )
        if isinstance(self.data_params, dataparams.ResolvedDataParams):
            check_fields(resolved_only_fields, self.data_params, datasets_list)
        else:
            check_fields(resolved_only_fields, datasets_list[0].data_params, datasets_list[1:])