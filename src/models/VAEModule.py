from typing import Any, Dict, Union, Optional

import torch
from torch import nn, Tensor
from torch.optim import lr_scheduler, Optimizer

# import lightning as l
# from lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import numpy as np
# import numpy.typing as npt

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class VAEModule(LightningModule):

    """
        LightningModule for Ecoacoustic VAE, the model has to be passed in only with optimiser and scheduler.
    """
    def __init__(
            self,
            model: nn.Module,
            optimizer: Optional[Optimizer]=None,
            scheduler: lr_scheduler=None,
            kl_scheduler: lr_scheduler=None,
            state_dict: Optional[Dict]=None,
            frozen: Optional[bool]=False,
            ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.model = model
        self.scheduler = scheduler
        self.kl_scheduler = kl_scheduler

        if state_dict:
            print("Using pretrained weights")
            self.load_state_dict(state_dict)
            if frozen:
                print("Freezing weights")
                # freeze teacher parameters
                for param in self.parameters():
                   param.requires_grad_(False)


    def forward(self, x: Tensor) -> Union[Tensor, Tensor, Tensor]:
        '''
            inference for the latent space
        '''
        return self.model.encode(x)

    def model_step(self, x: Tensor) -> Dict[str, Tensor]:

        # pass through the model outputs shape [batch_size,14,120,120]
        mu, logvar, z, recon, uncertainty = self.model(x)
        # reconstructed, uncertainty = self.model.decode(z)
        logvar = torch.zeros_like(logvar)

        if self.kl_scheduler:
            # Compute beta for cyclic annealing
            beta = self.kl_scheduler.scale
        else:
            beta = 1.0

        if self.trainer.global_step < 5000:
            uncertainty = torch.zeros_like(recon)

        # compute the reconstruction loss
        loss = self.model.loss_func(
                x,
                recon,
                mu,
                logvar,
                uncertainty,
                beta
                )

        return {**loss,
                'beta-kl': beta*loss['kl_divergence'].item(),
                'recon': recon.detach(),
                'x': x.detach(),
                'z': z.detach(),
                'uncertainty': uncertainty.detach()}


    def training_step(self,
                      batch: Dict[str, Tensor],
                      batch_idx: int) -> Dict[str, Tensor]:

        x, y = batch.values()

        outputs = self.model_step(x)

        return {**outputs,
                'y': y.detach()}

    def on_train_batch_end(self,
            outputs: Dict[str, Tensor],
            batch: Dict[str, Tensor],
            bathc_idx: int) -> None:

        self.kl_scheduler()

        # update and log metrics
        self.log_dict({"train/loss": outputs["loss"].item(),
                       "train/real_loss": outputs["real_loss"].item(),
                       "train/log_likelihood_of_data": -outputs["log_likelihood"].item(),
                       "train/kl_divergence": outputs["kl_divergence"].item(),
                       "train/beta_kl_divergence": outputs["beta-kl"]})


    def validation_step(self,
            batch: Dict[str, Tensor],
            batch_idx: int) -> Dict[str, Tensor]:

        x, y = batch.values()

        outputs = self.model_step(x)

        return {**outputs,
                'y': y.detach()}

    def on_validation_batch_end(self,
            outputs: Dict[str, Tensor],
            batch: Dict[str, Tensor],
            bathc_idx: int) -> None:

        # update and log metrics
        self.log_dict({"val/loss": outputs["loss"].item(),
                       "val/real_loss": outputs["real_loss"].item(),
                       "val/log_likelihood_of_data": -outputs["log_likelihood"].item(),
                       "val/kl_divergence": outputs["kl_divergence"].item()})


    def test_step(self, batch: Dict[str, Tensor], batch_idx: int):

        x, y = batch.values()

        outputs = self.model_step(x)

        return {**outputs,
                'y': y.detach()}

    def on_test_batch_end(self,
            outputs: Dict[str, Tensor],
            batch: Dict[str, Tensor],
            bathc_idx: int) -> None:

        # update and log metrics
        self.log_dict({"test/loss": outputs["loss"].item(),
                       "test/real_loss": outputs["real_loss"].item(),
                       "test/log_likelihood_of_data": -outputs["log_likelihood"].item(),
                       "test/kl_divergence": outputs["kl_divergence"].item() })

    def predict_step(self,
                     batch: Dict[str, Tensor],
                     batch_idx: int,
                     dataloader_idx: int) -> Dict[Any, Dict[str, np.ndarray]]:

        x, y = batch.values()

        # pass through the model
        output = self.model_step(x)

        return {**output,
                "y": y.detach()}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = self.hparams.optimizer(params=self.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
