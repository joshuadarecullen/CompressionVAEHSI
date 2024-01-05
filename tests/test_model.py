from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.bigearth_datamodule import BigEarthDataModule
# from src.models.VAEModule import VAEModule
from src.models.components.VAE import VAE
from src.models.components.encoder import Encoder
from src.models.components.decoder import Decoder
from src.models.components.lossfuncs import sam_loss

from src.callbacks.reconstructor import Reconstructor

# @pytest.mark.parametrize("batch_size", [32])
def test_model(batch_size: int) -> None:
    """Tests `model` to verify that the data flows through correctly, that the necessary
    attributes of the loss are the correct shape.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = BigEarthDataModule(
            dataset_dir=data_dir,
            train_val_test_split=(0.6, 0.2, 0.2),
            batch_size=batch_size,
            bands='all',
            )
    dm.setup()

    dm.setup_folds(5)
    dm.setup_fold_index(0)

    batch = next(iter(dm.train_dataloader()))
    x, y = batch.values()

    model = VAE(Encoder(), Decoder())

    mu, logvar, z, recon, uncertainty = model(x)

    kl = model.kl_divergence(mu, logvar).mean(axis=0)

    log_likelihood = F.gaussian_nll_loss(recon, x, uncertainty, reduction="none").view(x.size(0), -1).sum(axis=1).mean(axis=0)

    samL = sam_loss(x, recon)

    print(f"kl: {kl}")
    print(f"nll: {log_likelihood}")
    print(f"sam loss: {samL}")

    outputs = {"x": x,
               "recon": recon,
               "y": y,
               "uncertainty": uncertainty}

    reconstructor = Reconstructor(40, 0.25)
    reconstructor.generate_figures(outputs)


if __name__ == "__main__":
    test_model(32)
