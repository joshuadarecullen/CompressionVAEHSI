from typing import List, Dict, Any, Iterator

import wandb
from torch import Tensor
import lightning as L
from lightning import Callback
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class Reconstructor(Callback):
    def __init__(self,
                 reconstruct_step: int = 40,
                 perc_recon: float = 0.25
                 ) -> None:
        super().__init__()
        '''
        A callback to reconstruct the output of the VAE
        '''

        self.reconstruct_step = reconstruct_step # how often we recreate training reconstructions
        self.perc_recon = perc_recon # amount of reconstructions to produce per batch


    def on_train_batch_end(self,
                           trainer: L.Trainer,
                           pl_module: L.LightningModule,
                           outputs: Dict[str, Tensor],
                           batch: Dict[str, Tensor],
                           batch_idx: int) -> None:

        # recreate every 40 global steps
        if trainer.global_step % self.reconstruct_step == 0:
            # this import here stopped the psutil error, and tkinker error of image not bein gin main thread
            from matplotlib import pyplot as plt
            for fig in self.generate_figures(trainer, outputs):
                pl_module.logger.experiment.log({"train/image": wandb.Image(fig) })
                plt.close(fig)

    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: Dict[str, Tensor],
                                batch: Dict[str, Tensor],
                                batch_idx: int,
                                ) -> None:
        from matplotlib import pyplot as plt
        for fig in self.generate_figures(trainer ,outputs):
            pl_module.logger.experiment.log({"val/image": wandb.Image(fig) })
            plt.close(fig)

    def on_test_batch_end(self,
                          trainer: L.Trainer,
                          pl_module: L.LightningModule,
                          outputs: Dict[str, Tensor],
                          batch: Dict[str, Tensor],
                          batch_idx: int,
                          ) -> None:

        from matplotlib import pyplot as plt
        for fig in self.generate_figures(trainer ,outputs):
            pl_module.logger.experiment.log({"test/image": wandb.Image(fig) })
            plt.close(fig)


    def generate_figures(self,
                   trainer: L.Trainer,
                   outputs: Dict[str, Tensor]) -> Iterator[Figure]:

        # loading  modules here stops internel rm tree os error and tkinter main thread error
        from matplotlib import pyplot as plt
        from matplotlib import gridspec as gs
        import numpy as np

        # extract outputs required
        xs, recons, ys, x_stds = [outputs[key] for key in ["x", "recon", "y", "uncertainty"]]

        x_stds = (0.5 * outputs["uncertainty"]).exp()
        num_rows = 3
        mask = np.random.randint(0, xs.size(0), int(xs.size(0)*self.perc_recon)) # take a percentage of the batch

        # create and yield a figure for each observation
        for j, (x, recon, y, x_std) in enumerate(zip(xs[mask], recons[mask], ys[mask], x_stds[mask])):
            # create a figure and grid spec
            fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
            grid_spec = gs.GridSpec(num_rows, 2,
                                    width_ratios=[1, 0.04],
                                    hspace=0.8)

            # get min/max values for colourbar boundaries

            # plot observation, reconstruction and uncertainty image
            ax1 = fig.add_subplot(grid_spec[0, 0])
            ax2 = fig.add_subplot(grid_spec[1, 0])
            ax3 = fig.add_subplot(grid_spec[2, 0])

            # plot input
            ax1.set_title("Input HSI", fontsize='medium')
            x = self.correct_img(x[3:6,:,:].numpy())
            ax1.imshow(x)

            # plot reconstruction
            ax2.set_title("Reconstructed HSI", fontsize='medium')
            recon = self.correct_img(recon[3:6,:,:].numpy())
            ax2.imshow(recon)


            # plot uncertainty
            ax3.set_title("Uncertainty Spectrogram", fontsize='medium')
            x_std = self.correct_img(x_std[3:6,:,:].numpy())
            ax3.imshow(x_std)

            # set a title
            suptitle = ", ".join(self.get_label_names(y, trainer))
            fig.suptitle(suptitle, wrap=True)
            plt.show()

            # return fig

    def correct_img(self, img):
        img = self.normalise(img)
        img = np.moveaxis(img, 0, 2) # C, W, H > W, H, C
        img = img[:,:,::-1] # BGR to RGB
        return img

    def normalise(self, img):
        for band in range(len(img)):
            # normalize based on min/max pixel values to clamp ranges in [0, 1]
            img[band, ...] = (img[band, ...] - np.min(img[band, ...])) / np.max(img[band, ...])
        return img

    def get_label_names(self, onehots, class_names):
        """get all the names when a label is a one-hot."""
        label_idx = np.where(onehots)[0]
        label_names = [class_names[19][idx] for idx in label_idx]
        return label_names