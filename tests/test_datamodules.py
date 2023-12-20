from pathlib import Path

import pytest
import torch

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.datamodule import BigEarthDataModule


@pytest.mark.parametrize("batch_size", [32])
def test_bigearth_datamodule(batch_size: int) -> None:
    """Tests `BigEarthDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = BigEarthDataModule(
            dataset_dir=data_dir,
            train_val_test_split=(0.6, 0.2, 0.2),
            batch_size=batch_size
            )
    dm.prepare_data()

    # assert not dm.data_train and not dm.data_val and not dm.data_test
    # print(Path(data_dir, "master.csv"))
    assert Path(data_dir, "BigEarthNet-v1.0").exists() and Path(data_dir, "BigEarthNet-S1-v1.0").exists()
    assert Path(data_dir, "master.csv").exists()

    dm.setup()
    assert dm.train_dataset and dm.test_dataset

    dm.setup_folds(5)
    assert len(dm.splits) == 5

    dm.setup_fold_index(1)
    assert dm.train_fold and dm.val_fold
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.train_dataset) + len(dm.test_dataset) #+ len(dm.data_val)
    assert num_datapoints == 519285

    batch = next(iter(dm.train_dataloader()))
    x, y = batch.values()
    assert len(x) == batch_size
    assert len(y) == batch_size
    # assert x.dtype == torch.float32
    # assert y.dtype == torch.int64
