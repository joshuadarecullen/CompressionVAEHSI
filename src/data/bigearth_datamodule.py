from typing import Any, Dict, Optional, Tuple, AnyStr, List, Callable

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torch import Tensor
# from torchvision.transforms import transforms as Transform
from sklearn.model_selection import KFold
import kornia.augmentation as K
from torchgeo.transforms import AugmentationSequential

from torchgeo.datamodules.bigearthnet import BigEarthNetDataModule

import pathlib
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.base_kfoldmodule import BaseKFoldDataModule
from src.data.bigearth_dataset import BigEarthDataset


class BigEarthDataModule(BigEarthNetDataModule, BaseKFoldDataModule):
    """Data module class that prepares BigEarthNet-S2 dataset parsers and instantiates data loaders."""

    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        num_workers: int = 1,
        bands: str = "s2",
        pin_memory: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ):
        super().__init__(root_dir=dataset_dir,
                         batch_size=batch_size,
                         num_workers=num_workers)


        """Validates the hyperparameter config dictionary and sets up internal attributes."""
        self.save_hyperparameters(logger=False)

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.bands = bands
        self.train_split, self.valid_split, self.test_split = train_val_test_split
        self.norm_transforms = AugmentationSequential(K.Normalize(mean=self.mean, std=self.std), data_keys=['image'])
        self.transforms = transforms

        label_decoder: Dict[int, List[str]] = None
        label_converter: Dict[int,int] = None

        self.train_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        train_fold: Optional[Dataset] = None
        val_fold: Optional[Dataset] = None



    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes (19).
        """
        return len(self.train_dataset.class_names)

    def prepare_data(self) -> None:
        data = BigEarthDataset(
                root=self.dataset_dir,
                bands=self.bands,
                download=True,
                )

    def setup(self, stage: Optional[str] = None) -> None:
        """Parses and splits all samples across the train/valid/test datasets."""
        # self.dataset_path = download_data(self.dataset_dir, self.dataset_name)
        dataset = BigEarthDataset(
                                root=self.dataset_dir,
                                bands=self.bands,
                                transforms=self.transforms
                                )
        self.label_decoder = dataset.class_sets
        self.label_converter = dataset.label_converter

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.train_dataset and not self.test_dataset:

            self.train_dataset, self.test_dataset = random_split(dataset,
                                                                 [self.train_split+self.valid_split,
                                                                  self.test_split])

        # if self.train_dataset is None:
        #     self.train_dataset = BigEarthNetDataset(
        #         self.dataset_path / "train",
        #         transforms=self.transforms,
        #     )
        # if self.valid_dataset is None:
        #     self.valid_dataset = BigEarthNetDataset(
        #         self.dataset_path / "val",
        #         transforms=self.transforms,
        #     )
        # if self.test_dataset is None:
        #     self.test_dataset = BigEarthNetDataset(
        #         self.dataset_path / "test",
        #         transforms=self.transforms,
        #     )

    def train_dataloader(self) -> DataLoader:
        """Creates the training dataloader using the training dataset."""
        assert self.train_fold is not None, "must call 'setup() and 'setup_fold_index' first!"
        return DataLoader(
            dataset=self.train_fold,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the validation dataloader using the validation data parser."""
        assert self.val_fold is not None, "must call 'setup()' and 'setup_fold_index' first!"
        return DataLoader(
            dataset=self.val_fold,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates the testing dataloader using the testing data dataset."""
        assert self.test_dataset is not None, "must call 'setup' first!"
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.num_workers,
        )

    '''
    def on_before_batch_transfer(self, batch, dataloader_idx):
        # batch = {'image': batch['image'].squeeze(1), 'label': batch['label']}
        batch = self.norm_transforms(batch)
        # batch['x'] = transforms(batch['x'])
        return batch

    def on_after_batch_transfer(
         self, batch: Dict[str, Tensor], dataloader_idx: int
     ) -> Dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

         Args:
             batch: A batch of data that needs to be altered or augmented.
             dataloader_idx: The index of the dataloader to which the batch belongs.

         Returns:
             A batch of data.
         """
        if self.trainer:
            if self.trainer.training:
                split = "train"
            elif self.trainer.validating or self.trainer.sanity_checking:
                split = "val"
            elif self.trainer.testing:
                split = "test"
            elif self.trainer.predicting:
                split = "predict"

            # aug = self._valid_attribute(f"{split}_aug", "aug")
            # batch = aug(batch)
            batch = {'image': batch['image'].squeeze(1), 'label': batch['label']}
            batch = self.transforms(batch)

        return batch
    '''


    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

if __name__ == "__main__":
    _ = BigEarthNetDataModule(None, None)
