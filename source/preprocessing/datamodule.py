"""A custom pytorch-lightning datamodule for PurePlay."""

import lightning
import pydantic
import torch

from globals import processors
from misc import logging_utils
from . import dataset_builder, dataset, scalers, params

COMMON_DATALOADER_KWARGS = {
    "num_workers": processors.CPU_WORKERS,
    "pin_memory": True,
    "persistent_workers": True,
}

logger = logging_utils.get_logger()


class PurePlayDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        params: params.ProcessingParams | params.DataParams,
        batch_size: int = 32,
        train_dirs: dict[pydantic.DirectoryPath, int] = None,
        val_dirs: dict[pydantic.DirectoryPath, int] = None,
        test_dir: pydantic.DirectoryPath = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.params = params
        self.batch_size = batch_size
        self.train_dirs = train_dirs or {}
        self.val_dirs = val_dirs or {}
        self.test_dir = {test_dir: 0}

        self.scaler: scalers.SupportedScaler = scalers.SCALER_CACHE[0]
        self.builder = dataset_builder.DatasetBuilder(self)

        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets = []  # Kept separate for reporting

    def _all_training_files(self) -> list[pydantic.FilePath]:
        """Returns a list of all training file paths."""
        return [
            file_path
            for directory in self.train_dirs
            for file_path in self.builder.get_files_from_dir(directory)
        ]

    def setup(self, stage: str):
        if stage == "fit":
            if self.train_dataset and self.val_dataset:
                return

            logger.info("Fitting scaler on training files...")
            for file_path in self._all_training_files:
                data_frame = dataset.ParquetDataset.load_file(
                    file_path, self.params.whitelist
                )
                self.scaler.partial_fit(data_frame)

            logger.info("Creating datasets for training and validation files...")
            train_datasets = self.builder.make_datasets(self.train_dirs)
            val_datasets = self.builder.make_datasets(self.val_dirs)

            logger.info("Checking param consistency between datasets...")
            self.builder.check_consistency(train_datasets + val_datasets)

            logger.info("Concatenating datasets...")
            self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            self.val_dataset = torch.utils.data.ConcatDataset(val_datasets)

        if stage == "test":
            if self.test_datasets:
                return

            logger.info("Creating datasets for testing files...")
            self.test_datasets = self.builder.make_datasets(self.test_dir)

            logger.info("Checking param consistency between datasets...")
            self.builder.check_consistency(self.test_datasets)

        logger.info(f"Setup complete for stage: {stage}.")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            **COMMON_DATALOADER_KWARGS,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            **COMMON_DATALOADER_KWARGS,
        )

    def test_dataloader(self) -> list[torch.utils.data.DataLoader]:
        return [
            torch.utils.data.DataLoader(dataset=dataset, **COMMON_DATALOADER_KWARGS)
            for dataset in self.test_datasets
        ]
