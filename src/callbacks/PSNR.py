from typing import List, Dict, Any, Iterator

import wandb
import math

from torch import Tensor, log10, sqrt
import torch.nn.functional as F
import lightning as L
from lightning import Callback
import numpy as np
import matplotlib.pyplot as plt

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
        self.MAX_Is = {
                       0: 5,
                       1: 1,
                       2: 1,
                       3: 1,
                       4: 1,
                       5: 1,
                       6: 3,
                       7: 3,
                       8: 3,
                       9: 3,
                       10: 3,
                       11: 3,
                       12: 3,
                       13: 3
                       }

    
    def on_train_batch_end(self,
                           trainer: L.Trainer,
                           pl_module: L.LightningModule,
                           outputs: Dict[str, Tensor],
                           batch: Dict[str, Tensor],
                           batch_idx: int) -> None:

        psnrs = {}
        for j, (x, recon) in enumerate(zip(outputs['x'], outputs['recon'])):
            psnr = self.psnr(x, recon, j)
            psnrs[j] = psnr

        pl_module.log("train/band_psnr", psnrs)

    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: Dict[str, Tensor],
                                batch: Dict[str, Tensor],
                                batch_idx: int,
                                ) -> None:
        psnrs = {}
        for j, (x, recon) in enumerate(zip(outputs['x'], outputs['recon'])):
            psnr = self.psnr(x, recon, j)
            psnrs[j] = psnr

        pl_module.log("val/band_psnr", psnrs)

    def on_test_batch_end(self,
                          trainer: L.Trainer,
                          pl_module: L.LightningModule,
                          outputs: Dict[str, Tensor],
                          batch: Dict[str, Tensor],
                          batch_idx: int,
                          ) -> None:
        psnrs = {}
        for j, (x, recon) in enumerate(zip(outputs['x'], outputs['recon'])):
            psnr = self.psnr(x, recon, j)
            psnrs[j] = psnr

        pl_module.log("test/band_psnr", psnrs)

    def psnr(self,
             band1: Tensor,
             band2: Tensor,
             bandType: str) -> float:
        """
        Calculates the PSNR between two images.
        
        Args:
        band1 (torch.Tensor): The original image.
        band2 (torch.Tensor): The reconstructed image.

        Returns:
        float: The PSNR value.
        """
        mse = F.mse_loss(band1 - band2)
        if mse == 0:
            # If the images are identical, return an infinite PSNR
            return float('inf')
        
        MAX_I = self.MAX_Is[bandType]  # Assumes images are scaled between 0 and 1
        psnr_value = 20 * log10(MAX_I / sqrt(mse))
        
        return psnr_value.item()
