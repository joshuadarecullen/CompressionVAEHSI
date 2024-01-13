from typing import Dict

from torch import Tensor, log10, sqrt, inf
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer, LightningModule
import numpy as np

from matplotlib.figure import Figure

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class PSNR(Callback):
    def __init__(self):
        super().__init__()
        """
        A callback to compute the Peak Signal to Noise Ratio (PSNR)
        on the indvidual bands of the hyperspectral image
        """

    def on_train_batch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           outputs: Dict[str, Tensor],
                           batch: Dict[str, Tensor],
                           batch_idx: int) -> None:

        psnrs = self.generate_loss(outputs)

        pl_module.log_dict(psnrs)

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: Dict[str, Tensor],
                                batch: Dict[str, Tensor],
                                batch_idx: int,
                                ) -> None:

        psnrs = self.generate_loss(outputs)


        pl_module.log_dict(psnrs)

    def on_test_batch_end(self,
                          trainer: Trainer,
                          pl_module: LightningModule,
                          outputs: Dict[str, Tensor],
                          batch: Dict[str, Tensor],
                          batch_idx: int,
                          ) -> None:

        psnrs = self.generate_loss(outputs)

        pl_module.log_dict(psnrs)

    def generate_loss(self,
                      outputs: Dict[str,Tensor]) -> Dict[str,float]:
        psnrs = []
        for j, (x, recon) in enumerate(zip(outputs['x'], outputs['recon'])):
            band_psnrs = []

            for i, (x_band, recon_band) in enumerate(zip(x, recon)):
                psnr = self.psnr(x_band, recon_band)#, j, maxs)
                band_psnrs.append(psnr)


            psnrs.append(band_psnrs)
        psnr_means = np.mean(np.array(psnrs), axis=0)
        tagged_psnrs = {f'band_{i}': psnr_means[i] for i in range(len(psnr_means))}

        return tagged_psnrs

    def psnr(self,
             band1: Tensor,
             band2: Tensor) -> float:
        """
        Calculates the PSNR between two images.
        Args:
        band1 (torch.Tensor): The original image band.
        band2 (torch.Tensor): The reconstructed image band.

        Returns:
        float: The PSNR value.
        """
        mse = F.mse_loss(band1, band2)
        if not mse:
            # If the images are identical, return an infinite PSNR
            return inf
        
        MAX_I = 1.0 #MAX_Is[bandType]  # Assumes images are scaled between 0 and 1
        psnr_value = 20 * log10(MAX_I / sqrt(mse))
        
        return psnr_value.item()
