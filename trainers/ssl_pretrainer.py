from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
from utils.train_utils import masked_mse_loss, masked_mae_loss, gradient_loss, masked_min_max_loss, ccc_loss, auto_correlation_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from utils.plot_utils import plot_reconstruction, plot_generation
import numpy as np
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
import torch
import lightning
import sys
import math
from torch.optim.lr_scheduler import LambdaLR
from schedulers import get_cosine_with_hard_restarts_schedule_with_warmup_and_decay


# define the LightningModule
class PretrainedxLSTMNetwork(L.LightningModule):
    def __init__(
            self, 
            model, 
            len_train_dataset,
            config
        ):
        super().__init__()
        self.lr = config.lr
        self.model = model
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.wd = config.wd
        self.use_scheduler = config.use_scheduler
        self.patch_size = config.patch_size
        self.epochs = config.epochs
        self.loss_type = config.loss_type
        # self.config = config
        self.len_train_dataset = len_train_dataset
        self.num_epochs_warmup = config.num_epochs_warmup
        self.num_epochs_warm_restart = config.num_epochs_warm_restart
        self.sched_decay_factor = config.sched_decay_factor
        self.grad_loss_lambda = config.grad_loss_lambda
        self.min_max_loss_lambda = config.min_max_loss_lambda
        self.ccc_loss_lambda = config.ccc_loss_lambda
        self.auto_correlation_loss_lambda = config.auto_correlation_loss_lambda
        if not config.is_sweep:
            self.save_hyperparameters()

    def training_step(self, batch, _):
        loss = self.reconstruct_batch(batch, step='train')
        # Logging to TensorBoard (if installed) by default
        return loss
    
    def validation_step(self, batch, _):
        loss = self.reconstruct_batch(batch, step='val')
        return loss
    
    def test_step(self, batch, _):
        loss = self.reconstruct_batch(batch, step='test')
        return loss
    
    def on_train_epoch_end(self):
        """
        When the training loop ends, some representative plots from different classes are saved on wandb
        """
        sample_1 = self.trainer.train_dataloader.dataset[0]
        sample_2 = self.trainer.train_dataloader.dataset[-42]

        # get two random samples from the training dataset
        idx_3 = np.random.randint(0, len(self.trainer.train_dataloader.dataset))
        idx_4 = np.random.randint(0, len(self.trainer.train_dataloader.dataset))

        sample_3 = self.trainer.train_dataloader.dataset[idx_3]
        sample_4 = self.trainer.train_dataloader.dataset[idx_4]

        log_dir = self.logger.log_dir if self.logger.log_dir is not None else self.logger.experiment.dir

        img_1 = plot_reconstruction(sample_1, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_1')
        img_2 = plot_reconstruction(sample_2, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_2')
        img_3 = plot_reconstruction(sample_3, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_3_random')
        img_4 = plot_reconstruction(sample_4, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_4_random')
        if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.logger.log_image(key="reconstructions_train", images=[img_1, img_2, img_3, img_4])
        
        img_1 = plot_generation(sample_1, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_1')
        img_2 = plot_generation(sample_2, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_2')
        img_3 = plot_generation(sample_3, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_3_random')
        img_4 = plot_generation(sample_4, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_4_random')
        if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.logger.log_image(key="generations_train", images=[img_1, img_2, img_3, img_4])

        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        """
        When the validation loop ends, some representative plots from different classes are saved on wandb
        """
        # save the plots of the reconstruction for some samples
        sample_s = self.trainer.val_dataloaders.dataset[420]
        sample_v = self.trainer.val_dataloaders.dataset[1967]
        sample_t = self.trainer.val_dataloaders.dataset[4362]
        sample_n = self.trainer.val_dataloaders.dataset[0]

        log_dir = self.logger.log_dir if self.logger.log_dir is not None else self.logger.experiment.dir

        img_s = plot_reconstruction(sample_s, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_s')
        img_v = plot_reconstruction(sample_v, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_v')
        img_t = plot_reconstruction(sample_t, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_t')
        img_n = plot_reconstruction(sample_n, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_n')
        if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.logger.log_image(key="reconstructions", images=[img_s, img_v, img_t, img_n])

        img_s = plot_generation(sample_s, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_s')
        img_v = plot_generation(sample_v, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_v')
        img_t = plot_generation(sample_t, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_t')
        img_n = plot_generation(sample_n, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_n')
        if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.logger.log_image(key="generations", images=[img_s, img_v, img_t, img_n])
        return super().on_validation_epoch_end()
    
    def reconstruct_batch(self, batch, step):
        x = batch["signal"]
        mean_target = batch["r_peak_interval_mean"]
        var_target = batch["r_peak_variance"]

        # print('x shape', x.shape)

        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        tab_data = batch["tab_data"] if "tab_data" in batch.keys() else None

        x = F.pad(x, (0, 0, 0, self.patch_size - x.shape[1] % self.patch_size))

        reconstruction = self.model.reconstruct(x, tab_data)
        #print('reconstruction shape', reconstruction.shape)
        nrmse = np.inf

        shift_x = x[:, self.patch_size:].squeeze()
        shift_reconstruct = reconstruction[:, :-self.patch_size]
        #print('shift_reconstruct shape', shift_reconstruct.shape)

        # compute the loss and use the gradients only when it is needed
        if 'min_max' in self.loss_type:
            min_max = masked_min_max_loss(shift_reconstruct, shift_x, patch_size=self.patch_size)
       
        if 'ccc' in self.loss_type:
            ccc = ccc_loss(shift_reconstruct, shift_x)

        if 'ac' in self.loss_type:
            ac = auto_correlation_loss(shift_reconstruct, shift_x, convolution=self.model.patch_embedding)

        if 'mae' in self.loss_type:
            mae = (shift_reconstruct, shift_x)
        else:
            with torch.no_grad(): mae = masked_mae_loss(shift_reconstruct, shift_x)

        if 'grad' in self.loss_type:
            grad = gradient_loss(shift_reconstruct, shift_x)
        else:
            with torch.no_grad(): grad = gradient_loss(shift_reconstruct, shift_x)
        
        if 'mse' in self.loss_type:
            mse = masked_mse_loss(shift_reconstruct, shift_x, reduction='mean')
        else:
            with torch.no_grad(): mse = masked_mse_loss(shift_reconstruct, shift_x, reduction='mean')

        
   
        # calculate the normalized root squared error only for the first token prediction
        with torch.no_grad():
            nrmse = torch.sqrt(mse) / (shift_x.max() - shift_x.min())


        self.log(f"{step}_mse", mse.item(), prog_bar=False, batch_size=self.batch_size)
        self.log(f"{step}_mae", mae.item(), prog_bar=False, batch_size=self.batch_size)
        self.log(f"{step}_grad", grad.item(), prog_bar=False, batch_size=self.batch_size)
        if 'min_max' in self.loss_type: self.log(f"{step}_min_max", min_max.item(), prog_bar=False, batch_size=self.batch_size)
        if 'ccc' in self.loss_type: self.log(f"{step}_ccc", ccc.item(), prog_bar=True, batch_size=self.batch_size)
        if 'ac' in self.loss_type: self.log(f"{step}_ac", ac.item(), prog_bar=True, batch_size=self.batch_size)

        self.log(f"{step}_nrmse", nrmse.mean().item(), prog_bar=True, batch_size=self.batch_size)
        
        loss = torch.tensor(0.0, device=self.device)
        if 'mae' in self.loss_type: loss += mae
        elif 'mse' in self.loss_type: loss += mse

        if 'grad' in self.loss_type: loss += grad * self.grad_loss_lambda
        if 'min_max' in self.loss_type: loss += min_max * self.min_max_loss_lambda
        if 'ccc' in self.loss_type: loss += ccc * self.ccc_loss_lambda
        if 'ac' in self.loss_type: loss += ac * self.auto_correlation_loss_lambda

        self.log(f"{step}_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == 'adafactor':
            optimizer = optim.Adafactor(self.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)

        if self.use_scheduler:
            steps_per_epoch = np.ceil(self.len_train_dataset / self.batch_size)
            num_training_steps = steps_per_epoch * self.epochs
            warmup_steps = steps_per_epoch * self.num_epochs_warmup

            sched = get_cosine_with_hard_restarts_schedule_with_warmup_and_decay(
                optimizer, 
                num_warmup_steps = warmup_steps, 
                num_training_steps = num_training_steps, 
                num_cycles = (self.epochs - self.num_epochs_warmup) // self.num_epochs_warm_restart,
                decay_factor=self.sched_decay_factor
            )

            scheduler = {
                'scheduler': sched,
                'interval': 'step', # or 'epoch' 
                'frequency': 1,
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]

