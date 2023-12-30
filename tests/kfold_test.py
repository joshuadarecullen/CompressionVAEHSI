from pathlib import Path

import pytest
import torch

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.bigearth_datamodule import BigEarthDataModule
from src.data.kfoldloop import KFoldLoop
from src.models.VAEModule import VAEModule
from src.models.components.VAE import VAE
from src.models.components.encoder import Encoder
from src.models.components.decoder import Decoder
from pytorch_lightning import Trainer

# @pytest.mark.parametrize("batch_size", [32])
def test_bigearth_datamodule(batch_size: int) -> None:
    """Tests `BigEarthDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes, batch sizes, and bands
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """


if __name__ == "__main__":
    data_dir = "data/"
    model = VAEModule(model=VAE(Encoder(), Decoder()))
    model.optimizer=torch.optim.Adam(lr=0.001, params=model.parameters())

    dm = BigEarthDataModule(
            dataset_dir=data_dir,
            train_val_test_split=(0.6, 0.2, 0.2),
            batch_size=16,
            bands='all',
            )
    # dm.setup()

    trainer = Trainer(
        max_epochs=10,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        num_sanity_val_steps=0,
        devices=1,
        accelerator="auto",
        # strategy="ddp",
    )
    trainer.fit_loop = KFoldLoop(5, trainer.fit_loop, trainer, export_path="./")
    trainer.fit(model, dm)
