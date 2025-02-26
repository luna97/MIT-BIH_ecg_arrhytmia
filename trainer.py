from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
from utils.train_utils import masked_mse_loss, masked_mae_loss, gradient_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from plot_utils import plot_reconstruction, plot_generation
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
        self.multi_token_prediction = config.multi_token_prediction
        self.epochs = config.epochs
        self.loss_type = config.loss_type
        # self.config = config
        self.len_train_dataset = len_train_dataset
        self.num_epochs_warmup = config.num_epochs_warmup
        self.num_epochs_warm_restart = config.num_epochs_warm_restart
        self.sched_decay_factor = config.sched_decay_factor
        if not config.is_sweep:
            self.save_hyperparameters()

    def training_step(self, batch, _):
        loss = self.reconstruct_batch(batch, step='train')
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, _):
        loss = self.reconstruct_batch(batch, step='val')
        self.log("val_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch, _):
        loss = self.reconstruct_batch(batch, step='test')
        self.log("test_mse", loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss

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
        mask = batch["mask"]
        tab_data = batch["tab_data"] if "tab_data" in batch.keys() else None

        x = F.pad(x, (0, 0, 0, self.patch_size - x.shape[1] % self.patch_size))
        mask = F.pad(mask, (0, 0, 0, self.patch_size - mask.shape[1] % self.patch_size))

        reconstruction = self.model.reconstruct(x, tab_data)

        if self.multi_token_prediction:
            # tokens as array from uple
            tokens = [r1, r2, r3, r4] = reconstruction
        else:
            tokens = [r]

        maes, mses, grads = [], [], []
        nrmse = np.inf

        for i, r in enumerate(tokens, start=1):
            shift_x = x[:, self.patch_size * i:].squeeze()
            mask_shifted = mask[:, self.patch_size * i:].squeeze()
            shift_reconstruct = r[:, :-self.patch_size * i]

            # compute the loss and use the gradients only when it is needed
            if self.loss_type == 'mae':
                maes.append(masked_mae_loss(shift_reconstruct, shift_x, mask=mask_shifted))
            else:
                with torch.no_grad(): maes.append(masked_mae_loss(shift_reconstruct, shift_x, mask=mask_shifted))

            if self.loss_type == 'grad':
                grads.append(gradient_loss(shift_reconstruct, shift_x, mask=mask_shifted))
            else:
                with torch.no_grad(): grads.append(gradient_loss(shift_reconstruct, shift_x, mask=mask_shifted))
            
            if self.loss_type == 'mse' or self.loss_type == 'grad':
                mse = masked_mse_loss(shift_reconstruct, shift_x, mask=mask_shifted, reduction='None')
            else:
                with torch.no_grad(): mse = masked_mse_loss(shift_reconstruct, shift_x, mask=mask_shifted, reduction='None')
            
            mses.append(mse.mean())

            if i == 1:
            # calculate the normalized root squared error only for the first token prediction
                with torch.no_grad():
                    nrmse = torch.sqrt(mse.mean(dim=0)) / (shift_x.max() - shift_x.min())

        mae = sum(maes) / len(maes)
        mse = sum(mses) / len(mses)
        grad = sum(grads) / len(grads)


        self.log(f"{step}_mse", mse.item(), prog_bar=True, batch_size=self.batch_size)
        self.log(f"{step}_mae", mae.item(), prog_bar=True, batch_size=self.batch_size)
        self.log(f"{step}_grad", grad.item(), prog_bar=True, batch_size=self.batch_size)
        self.log(f"{step}_nrmse", nrmse.mean().item(), prog_bar=True, batch_size=self.batch_size)
        
        if self.loss_type == 'mae':
            return mae
        elif self.loss_type == 'grad':
            return grad + mse 
        else:
            return mse

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
            # sched = get_cosine_with_hard_restarts_schedule_with_warmup(
            #    optimizer, 
            #    num_warmup_steps=steps_per_epoch * self.num_epochs_warmup, 
            #    num_training_steps=num_training_steps, 
            #    num_cycles=(num_training_steps // steps_per_epoch) // self.num_epochs_warm_restart)
            sched = get_cosine_with_hard_restarts_schedule_with_warmup_and_decay(
                optimizer, 
                num_warmup_steps = warmup_steps, 
                num_training_steps = num_training_steps, 
                num_cycles = (num_training_steps // warmup_steps) // self.num_epochs_warm_restart,
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


class TrainingxLSTMNetwork(L.LightningModule):
    def __init__(self, model, lr_head=1e-3, lr_xlstm=1e-4, batch_size=32, optimizer='adam', num_classes=5, wd=0.0001, weights=None):
        super().__init__()
        self.lr_head = lr_head
        self.lr_xlstm = lr_xlstm
        self.wd = wd
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.weights = weights
        self.save_hyperparameters()
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes)
        self.valid_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes)
        self.sen_cl1 = torchmetrics.classification.MulticlassSpecificity(num_classes=num_classes)

    def training_step(self, batch, _):
        loss, preds = self.predict_batch(batch)
        self.train_acc(preds, batch['label'])
        self.train_f1(preds, batch['label'])
        self.log('train_loss', loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log('train_acc', self.train_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('train_f1', self.train_f1, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, _):
        loss, preds = self.predict_batch(batch)
        self.valid_acc(preds, batch['label'])
        self.valid_f1(preds, batch['label'])
        self.log('val_loss', loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', self.valid_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('val_f1', self.valid_f1, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch, _):
        loss, preds = self.predict_batch(batch)
        self.test_acc(preds, batch['label'])
        self.test_f1(preds, batch['label'])
        self.log("test_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log("test_acc", self.test_acc, prog_bar=True, batch_size=self.batch_size)
        self.log("test_f1", self.test_f1, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def predict_batch(self, batch):
        ctx = batch["signal"]
        x = batch['heartbeat']
        targets = batch['label']
        preds = self.model(ctx, x)
        loss = nn.functional.cross_entropy(preds, targets, weight=self.weights)
        return loss, preds

    def get_params(self):
        return [
            {'params': self.model.fc.parameters(), 'lr': self.lr_head, 'weight_decay': self.wd},
            {'params': self.model.sep_token, 'lr': self.lr_head, 'weight_decay': self.wd},
            # {'params': self.model.xlstm.parameters(), 'lr': self.lr_xlstm, 'weight_decay': self.wd}
        ]
        
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(params=self.get_params())
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(self.get_params())
        else:
            optimizer = optim.SGD(self.get_params())
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {'optimizer': optimizer } # , 'lr_scheduler': scheduler }
