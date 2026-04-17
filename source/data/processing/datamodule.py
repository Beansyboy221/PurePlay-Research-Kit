import lightning
import pydantic
import torch

from globals import (
    processors,
    logger
)
from . import (
    dataset_builder,
    scaler_manager,
    data_params
)

COMMON_DATALOADER_KWARGS = {
    'num_workers': processors.CPU_WORKERS,
    'pin_memory': True,
    'persistent_workers': True
}

class PurePlayDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            data_params: data_params.ProcessingParams | data_params.DataParams,
            batch_size: int = 32,
            labeled_train_dirs: dict[pydantic.DirectoryPath, int] = None,
            labeled_validation_dirs: dict[pydantic.DirectoryPath, int] = None,
            testing_dir: pydantic.DirectoryPath = None
        ):
        super().__init__()
        self.data_params = data_params
        self.batch_size = batch_size
        self.labeled_train_dirs = labeled_train_dirs or {}
        self.labeled_validation_dirs = labeled_validation_dirs or {}
        self.testing_dir = testing_dir

        self.scaler_manager = scaler_manager.ScalerManager(data_params)
        self.builder = dataset_builder.DatasetBuilder(data_params, self.scaler_manager)

        self.train_dataset = None
        self.validation_dataset = None
        self.test_datasets = []
    
    def _all_training_files(self) -> list[pydantic.FilePath]:
        '''Returns a list of all training file paths.'''
        return [
            file_path
            for directory in self.labeled_train_dirs
            for file_path in self.builder.get_files_from_dir(directory)
        ]

    def setup(self, stage: str):
        if stage == 'fit':
            if self.train_dataset is not None or self.validation_dataset is not None:
                return

            logger.info(f'Fitting scaler on training files...')
            self.scaler_manager.fit(self._all_training_files())

            train_datasets = self.builder.make_datasets_from_labeled_dirs(self.labeled_train_dirs)
            validation_datasets = self.builder.make_datasets_from_labeled_dirs(self.labeled_validation_dirs)
            self.builder.check_consistency(train_datasets + validation_datasets)

            self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            self.validation_dataset = torch.utils.data.ConcatDataset(validation_datasets)

        if stage == 'test':
            if self.test_datasets:
                return

            self.test_datasets = self.builder.make_datasets(
                file_paths=self.builder.get_files_from_dir(self.testing_dir),
                label=0
            )
            self.builder.check_consistency(self.test_datasets)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            **COMMON_DATALOADER_KWARGS
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.batch_size,
            **COMMON_DATALOADER_KWARGS
        )

    def test_dataloader(self) -> list[torch.utils.data.DataLoader]:
        return [
            torch.utils.data.DataLoader(
                dataset=dataset,
                **COMMON_DATALOADER_KWARGS
            )
            for dataset in self.test_datasets
        ]