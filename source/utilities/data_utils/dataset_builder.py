import sklearn.utils.validation
import pydantic
import os

from . import (
    scaler_manager,
    data_params,
    dataset
	)

class DatasetBuilder:
    def __init__(
            self,
            data_params: data_params.DataParams,
            scaler_manager: scaler_manager.ScalerManager
        ):
        self.data_params = data_params
        self.scaler_manager = scaler_manager

    def get_files_from_dir(self, directory: str) -> list[pydantic.FilePath]:
        '''Returns a list of paths for all parquet files in the given directory.'''
        return [
            pydantic.FilePath(os.path.join(directory, file_path))
            for file_path in os.listdir(directory)
            if os.path.splitext(file_path)[1].lower() == '.parquet'
        ]

    def make_datasets(self, file_paths: list[str], label: int) -> list[dataset.FileDataset]:
        '''Creates FileDataset instances for each given file path.'''
        sklearn.utils.validation.check_is_fitted(
            estimator=self.scaler_manager.scaler,
            msg='Scaler must be fitted before creating datasets.'
        )
        return [
            dataset.FileDataset(
                file_path=file_path,
                data_params=self.data_params,
                scaler=self.scaler_manager.scaler,
                label=label
            )
            for file_path in file_paths
        ]

    def make_datasets_from_labeled_dirs(
            self,
            labeled_dirs: dict[pydantic.DirectoryPath, int]
        ) -> list[dataset.FileDataset]:
        '''Creates datasets for all directories, assigning each its corresponding label.'''
        result = []
        for directory, label in labeled_dirs.items():
            result += self.make_datasets(self.get_files_from_dir(directory), label=label)
        return result

    def check_consistency(self, datasets_list: list[dataset.FileDataset]) -> None:
        '''Checks that all dataset params match.'''
        if not datasets_list:
            return

        def check_fields(fields, reference_params, check_datasets):
            for field in fields:
                reference_value = getattr(reference_params, field)
                for dataset in check_datasets:
                    if getattr(dataset.data_params, field) != reference_value:
                        raise ValueError(f'Mismatched property: {field} found in: {dataset.file_path}')

        check_fields(data_params.DataParams.model_fields, self.data_params, datasets_list)

        resolved_only_fields = (
            data_params.ResolvedDataParams.model_fields.keys() 
            - data_params.DataParams.model_fields.keys()
        )
        if isinstance(self.data_params, data_params.ResolvedDataParams):
            check_fields(resolved_only_fields, self.data_params, datasets_list)
        else:
            check_fields(resolved_only_fields, datasets_list[0].data_params, datasets_list[1:])