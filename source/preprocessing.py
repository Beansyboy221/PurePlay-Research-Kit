import sklearn.preprocessing
import pyarrow.parquet
import lightning
import pydantic
import typing
import polars
import torch
import numpy
import os
import constants, devices, logger

class DataConfig(pydantic.BaseModel):
    """Parameters defining dataset properties."""
    whitelist: typing.List[str]
    polling_rate: int
    ignore_empty_polls: bool
    reset_mouse_on_release: bool
    polls_per_sequence: int

    @property
    def features_per_poll(self) -> int:
        return len(self.whitelist)

class InputDataset(torch.utils.data.Dataset):
    """Dataset for loading Parquet files."""
    def __init__(
            self,
            file_path: str,
            data_params: DataConfig,
            scaler: object,
            label: int = 0
        ):
        self.data_params = data_params
        self.label = label
        self.polling_rate = pyarrow.parquet.read_metadata(file_path).metadata.get(b'polling_rate').decode('utf-8')
        if not self.polling_rate:
            raise ValueError(f'Polling rate metadata is missing from file: {file_path}')
        
        # Filter out empty polls if needed (should I filter out whole sequences to preserve temporal structure?)
        lazy_frame = polars.scan_parquet(file_path).select(data_params.whitelist)
        if data_params.ignore_empty_polls:
            lazy_frame = lazy_frame.filter(polars.sum_horizontal(data_params.whitelist) != 0)
        data_array = lazy_frame.collect().to_numpy(writable=True).astype(numpy.float32)
        
        # Scale the data
        data_array = scaler.transform(data_array)
        
        # Trim excess rows to make full sequences
        num_sequences = len(data_array) // data_params.polls_per_sequence
        total_rows = num_sequences * data_params.polls_per_sequence
        data_array = data_array[:total_rows].reshape(num_sequences, data_params.polls_per_sequence, -1)

        self.data_tensor = torch.from_numpy(data_array)

    def __len__(self) -> int:
        """Returns the number of sequences in the dataset."""
        return len(self.data_tensor)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a sequence and its label."""
        return self.data_tensor[index], torch.tensor(self.label, dtype=torch.float32)

class PurePlayDataModule(lightning.LightningDataModule):
    def __init__(self, config: object, data_params: DataConfig = None):
        super().__init__()
        self.config = config
        self.batch_size = config.sequences_per_batch # Needs to be pulled out for tuning
        self.data_params = data_params or DataConfig(
            whitelist=config.keyboard_whitelist + config.mouse_whitelist + config.gamepad_whitelist,
            polling_rate=None,
            ignore_empty_polls=config.ignore_empty_polls, # Anything left as none here will be pulled from the data
            reset_mouse_on_release=None,
            polls_per_sequence=config.polls_per_sequence
        ).model_dump()

        self.scaler = sklearn.preprocessing.RobustScaler()
        
        self.train_dataset = None
        self.validation_dataset = None
        self.test_datasets = []

    def _get_files_from_dir(self, directory: str) -> list[str]:
        """Returns a list of file paths for all Parquet files in the given directory."""
        return [os.path.join(directory, file_path) for file_path in os.listdir(directory)]

    def _make_datasets(self, file_paths: list, label: int = 0) -> list[InputDataset]:
        """Creates InputDataset instances for each given file path."""
        return [InputDataset(
            file_path=file_path,
            data_params=self.data_params,
            scaler=self.scaler,
            label=label
        ) for file_path in file_paths]
    
    def _check_consistency(self, datasets: list[InputDataset]) -> None:
        """Checks that all dataset properties match."""
        for key, value in self.data_params:
            if value is None:
                value = datasets[0].data_params[key]
            for dataset in datasets:
                if dataset.data_params[key] != value:
                    raise ValueError(f'Mismatched parameter "{key}" found in "{dataset.file_path}"')

    def _fit_scaler_to_files(self, file_paths: list[str]) -> None:
        """Fits the internal scaler to a specified list of files."""
        logger.info(f'Fitting scaler on {len(file_paths)} files...')
        for file_path in file_paths:
            lazy_frame = polars.scan_parquet(file_path).select(self.whitelist)
            if self.data_params.ignore_empty_polls:
                lazy_frame = lazy_frame.filter(polars.sum_horizontal(self.whitelist) != 0)
            self.scaler.partial_fit(lazy_frame.collect().to_numpy().astype(numpy.float32))

    def update_scaler(self, scaler_name: str, scaler_params: dict = None) -> None:
        """Sets scaler class and optionally fits it to specified params."""
        self.scaler = constants.SCALER_MAP[scaler_name]
        if self.train_dataset and not scaler_params:
            all_training_files = self._get_files_from_dir(self.config.training_file_dir)
            if self.config.model_class.training_type == constants.TrainingType.SUPERVISED:
                all_training_files += self._get_files_from_dir(self.config.cheat_training_file_dir)
            self._fit_scaler_to_files(all_training_files)
        for attribute_name, value in scaler_params:
            if hasattr(self.scaler, attribute_name):
                setattr(self.scaler, attribute_name, value)

    def setup(self, stage: str = None) -> None:
        """Sets up the datasets for training, validation, or testing."""
        if stage == 'fit':
            if self.train_dataset is not None:
                return

            benign_training_files = self._get_files_from_dir(self.config.training_file_dir)
            all_training_files = benign_training_files.copy()
            train_datasets = self._make_datasets(benign_training_files)
            validation_datasets = self._make_datasets(self._get_files_from_dir(self.config.validation_file_dir))
            if self.config.model_class.training_type == constants.TrainingType.SUPERVISED:
                cheat_training_files = list(self._get_files_from_dir(self.config.cheat_training_file_dir))
                all_training_files.extend(cheat_training_files)
                train_datasets += self._make_datasets(cheat_training_files, label=1)
                validation_datasets += self._make_datasets(self._get_files_from_dir(self.config.cheat_validation_file_dir), label=1)

            self._fit_scaler_to_files(all_training_files)

            all_datasets = train_datasets + validation_datasets
            self._check_consistency(all_datasets)

            self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            self.validation_dataset = torch.utils.data.ConcatDataset(validation_datasets)

        if stage == 'test':
            if self.test_datasets is not None:
                return
            
            self.test_datasets = self._make_datasets(self._get_files_from_dir(self.config.testing_file_dir))
            self._check_consistency(self.test_datasets)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a DataLoader for the training dataset."""
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset, 
            shuffle=True, 
            batch_size=self.batch_size,
            num_workers=devices.CPU_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a DataLoader for the validation dataset."""
        return torch.utils.data.DataLoader(
            dataset=self.validation_dataset, 
            shuffle=False, 
            batch_size=self.batch_size,
            num_workers=devices.CPU_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self) -> list[torch.utils.data.DataLoader]:
        """Returns a list of DataLoaders for each test dataset."""
        return [
            torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                num_workers=devices.CPU_WORKERS,
                pin_memory=True,
                persistent_workers=True
            ) for dataset in self.test_datasets
        ]