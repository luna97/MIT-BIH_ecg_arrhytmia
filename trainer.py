from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
from utils.train_utils import masked_mse_loss, masked_mae_loss, gradient_loss, masked_min_max_loss
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
    
    def on_train_epoch_end(self):
        """
        When the training loop ends, some representative plots from different classes are saved on wandb
        """
        sample_1 = self.trainer.train_dataloader.dataset[0]
        sample_2 = self.trainer.train_dataloader.dataset[87]

        log_dir = self.logger.log_dir if self.logger.log_dir is not None else self.logger.experiment.dir

        img_1 = plot_reconstruction(sample_1, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_1')
        img_2 = plot_reconstruction(sample_2, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_2')
        if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.logger.log_image(key="reconstructions_train", images=[img_1, img_2])
        
        img_1 = plot_generation(sample_1, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_1')
        img_2 = plot_generation(sample_2, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_2')
        if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.logger.log_image(key="generations_train", images=[img_1, img_2])

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

        # print('x shape', x.shape)

        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        tab_data = batch["tab_data"] if "tab_data" in batch.keys() else None

        x = F.pad(x, (0, 0, 0, self.patch_size - x.shape[1] % self.patch_size))

        reconstruction = self.model.reconstruct(x, tab_data)

        if self.multi_token_prediction:
            # tokens as array from uple
            tokens = [r1, r2, r3, r4] = reconstruction
        else:
            tokens = [reconstruction]

        maes, mses, grads, min_max = [], [], [], []
        nrmse = np.inf

        for i, r in enumerate(tokens, start=1):
            shift_x = x[:, self.patch_size * i:].squeeze()
            shift_reconstruct = r[:, :-self.patch_size * i]

            # print('shift_x shape', shift_x.shape)
            # print('shift_reconstruct shape', shift_reconstruct.shape)

            if self.loss_type in ['min_max', 'grad_min_max']:
                min_max.append(masked_min_max_loss(shift_reconstruct, shift_x, patch_size=self.patch_size))
            else:
                with torch.no_grad(): min_max.append(masked_min_max_loss(shift_reconstruct, shift_x, patch_size=self.patch_size))

            # compute the loss and use the gradients only when it is needed
            if self.loss_type == 'mae':
                maes.append(masked_mae_loss(shift_reconstruct, shift_x))
            else:
                with torch.no_grad(): maes.append(masked_mae_loss(shift_reconstruct, shift_x))

            if self.loss_type in ['grad', 'grad_min_max']:
                grads.append(gradient_loss(shift_reconstruct, shift_x))
            else:
                with torch.no_grad(): grads.append(gradient_loss(shift_reconstruct, shift_x))
            
            if self.loss_type in ['mse', 'grad', 'min_max', 'grad_min_max']:
                mse = masked_mse_loss(shift_reconstruct, shift_x, reduction='none')
            else:
                with torch.no_grad(): mse = masked_mse_loss(shift_reconstruct, shift_x, reduction='none')
            
            mses.append(mse.mean())

            if i == 1:
            # calculate the normalized root squared error only for the first token prediction
                with torch.no_grad():
                    nrmse = torch.sqrt(mse.mean(dim=0)) / (shift_x.max() - shift_x.min())

        mae = sum(maes) / len(maes)
        mse = sum(mses) / len(mses)
        grad = sum(grads) / len(grads)
        min_max = sum(min_max) / len(min_max)


        self.log(f"{step}_mse", mse.item(), prog_bar=True, batch_size=self.batch_size)
        self.log(f"{step}_mae", mae.item(), prog_bar=True, batch_size=self.batch_size)
        self.log(f"{step}_grad", grad.item(), prog_bar=True, batch_size=self.batch_size)
        self.log(f"{step}_min_max", min_max.item(), prog_bar=True, batch_size=self.batch_size)
        self.log(f"{step}_nrmse", nrmse.mean().item(), prog_bar=True, batch_size=self.batch_size)
        
        if self.loss_type == 'mae': return mae
        if self.loss_type == 'mse':  return mse
        elif self.loss_type == 'grad': return grad + mse 
        if self.loss_type == 'min_max': return min_max  + mse
        if self.loss_type == 'grad_min_max': return grad + min_max + mse
        else: raise ValueError(f"Invalid loss type {self.loss_type}")

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
    def __init__(self, model, config,  len_train_dataset, num_classes=5, weights=None):
        super().__init__()
        self.lr_head = config.lr_head
        self.lr_xlstm = config.lr_xlstm
        self.wd = config.wd
        self.model = model
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.weights = weights
        self.use_scheduler = config.use_scheduler
        self.len_train_dataset = len_train_dataset
        self.num_epochs_warmup = config.num_epochs_warmup
        self.sched_decay_factor = config.sched_decay_factor
        self.num_epochs_warm_restart = config.num_epochs_warm_restart
        self.epochs = config.epochs
        self.save_hyperparameters()
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average='macro')
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average='macro')
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average='macro')
        self.train_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, top_k=1, average='macro')
        self.valid_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, top_k=1, average='macro')
        self.test_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, top_k=1, average='macro')
        # add sensitivity and specificity for the first class
        self.val_spec = torchmetrics.classification.specificity.MulticlassSpecificity(num_classes=num_classes, average=None)
        self.test_spec = torchmetrics.classification.specificity.MulticlassSpecificity(num_classes=num_classes, average=None)
        self.val_recall = torchmetrics.classification.precision_recall.MulticlassRecall(num_classes=num_classes, average=None)
        self.test_recall = torchmetrics.classification.precision_recall.MulticlassRecall(num_classes=num_classes, average=None)
        self.val_precision = torchmetrics.classification.precision_recall.MulticlassPrecision(num_classes=num_classes, average=None)
        self.test_precision = torchmetrics.classification.precision_recall.MulticlassPrecision(num_classes=num_classes, average=None)


    def training_step(self, batch, _):
        loss, _, preds = self.predict_batch(batch)
        self.train_acc(preds, batch['label'])
        self.train_f1(preds, batch['label'])
        self.log('train_loss', loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log('train_acc', self.train_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('train_f1', self.train_f1, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, _):
        loss, _, preds = self.predict_batch(batch)
        self.valid_acc(preds, batch['label'])
        self.valid_f1(preds, batch['label'])
        self.log('val_loss', loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', self.valid_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('val_f1', self.valid_f1, prog_bar=True, batch_size=self.batch_size)

        # specificity
        self.val_spec = self.val_spec.to(preds.device)
        self.val_spec(preds, batch['label'])
        self.log('val_specificity/N', self.val_spec[0], prog_bar=True, batch_size=self.batch_size, metric_attribute='val_spec')
        self.log('val_specificity/S', self.val_spec[1], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_spec')
        self.log('val_specificity/V', self.val_spec[2], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_spec')
        self.log('val_specificity/F', self.val_spec[3], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_spec')
        self.log('val_specificity/Q', self.val_spec[4], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_spec')

        # sensitivity
        self.val_recall = self.val_recall.to(preds.device)
        self.val_recall(preds, batch['label'])
        self.log('val_sensitivity/N', self.val_recall[0], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_recall')
        self.log('val_sensitivity/S', self.val_recall[1], prog_bar=True, batch_size=self.batch_size, metric_attribute='val_recall')
        self.log('val_sensitivity/V', self.val_recall[2], prog_bar=True, batch_size=self.batch_size, metric_attribute='val_recall')
        self.log('val_sensitivity/F', self.val_recall[3], prog_bar=True, batch_size=self.batch_size, metric_attribute='val_recall')
        self.log('val_sensitivity/Q', self.val_recall[4], prog_bar=True, batch_size=self.batch_size, metric_attribute='val_recall')

        # ppv
        self.val_precision = self.val_precision.to(preds.device)
        self.val_precision(preds, batch['label'])
        self.log('val_ppv/N', self.val_precision[0], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_precision')
        self.log('val_ppv/S', self.val_precision[1], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_precision')
        self.log('val_ppv/V', self.val_precision[2], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_precision')
        self.log('val_ppv/F', self.val_precision[3], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_precision')
        self.log('val_ppv/Q', self.val_precision[4], prog_bar=False, batch_size=self.batch_size, metric_attribute='val_precision')
        return loss
    
    def test_step(self, batch, _):
        loss, _, preds = self.predict_batch(batch)
        self.test_acc(preds, batch['label'])
        self.test_f1(preds, batch['label'])
        self.log("test_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log("test_acc", self.test_acc, prog_bar=True, batch_size=self.batch_size)
        self.log("test_f1", self.test_f1, prog_bar=True, batch_size=self.batch_size)

        # sensitivity
        self.test_spec = self.test_spec.to(preds.device)
        self.test_spec(preds, batch['label'])
        self.log("test_specificity/N", self.test_spec[0], batch_size=self.batch_size, metric_attribute='test_spec')
        self.log("test_specificity/S", self.test_spec[1], batch_size=self.batch_size, metric_attribute='test_spec')
        self.log("test_specificity/V", self.test_spec[2], batch_size=self.batch_size, metric_attribute='test_spec')
        self.log("test_specificity/F", self.test_spec[3], batch_size=self.batch_size, metric_attribute='test_spec')
        self.log("test_specificity/Q", self.test_spec[4], batch_size=self.batch_size, metric_attribute='test_spec')

        # sensitivity
        self.test_recall = self.test_recall.to(preds.device)
        self.test_recall(preds, batch['label'])
        self.log("test_sensitivity/N", self.test_recall[0], batch_size=self.batch_size, metric_attribute='test_recall')
        self.log("test_sensitivity/S", self.test_recall[1], batch_size=self.batch_size, metric_attribute='test_recall')
        self.log("test_sensitivity/V", self.test_recall[2], batch_size=self.batch_size, metric_attribute='test_recall')
        self.log("test_sensitivity/F", self.test_recall[3], batch_size=self.batch_size, metric_attribute='test_recall')
        self.log("test_sensitivity/Q", self.test_recall[4], batch_size=self.batch_size, metric_attribute='test_recall')

        # ppv
        self.test_precision = self.test_precision.to(preds.device)
        self.test_precision(preds, batch['label'])
        self.log("test_ppv/N", self.test_precision[0], batch_size=self.batch_size, metric_attribute='test_precision')
        self.log("test_ppv/S", self.test_precision[1], batch_size=self.batch_size, metric_attribute='test_precision')
        self.log("test_ppv/V", self.test_precision[2], batch_size=self.batch_size, metric_attribute='test_precision')
        self.log("test_ppv/F", self.test_precision[3], batch_size=self.batch_size, metric_attribute='test_precision')
        self.log("test_ppv/Q", self.test_precision[4], batch_size=self.batch_size, metric_attribute='test_precision')

        return loss
            
    
    def predict_batch(self, batch):
        ctx = batch["signal"]
        x = batch['heartbeat']
        tab_data = batch['tab_data'] if 'tab_data' in batch.keys() else None 
        targets = batch['label']
        out = self.model(ctx, x, tab_data)
        preds = torch.argmax(out, dim=1)
        loss = nn.functional.cross_entropy(out, targets, weight=self.weights)
        return loss, out, preds

    def get_params(self):
        return [
            # head and sep token with normal lr
            {'params': self.model.fc.parameters(), 'lr': self.lr_head, 'weight_decay': self.wd},
            {'params': self.model.sep_token, 'lr': self.lr_head, 'weight_decay': self.wd},

            # xlstm and patch embedding with lower lr
            {'params': self.model.xlstm.parameters(), 'lr': self.lr_xlstm, 'weight_decay': self.wd},
            {'params': self.model.patch_embedding.parameters(), 'lr': self.lr_xlstm, 'weight_decay': self.wd}
        ]
        
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(params=self.get_params())
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(self.get_params())
        elif self.optimizer == 'adafactor':
            optimizer = optim.Adafactor(self.get_params())
        else:
            optimizer = optim.SGD(self.get_params())

        if self.use_scheduler: 
            steps_per_epoch = np.ceil(self.len_train_dataset / self.batch_size)
            num_training_steps = steps_per_epoch * self.epochs
            warmup_steps = steps_per_epoch * self.num_epochs_warmup

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
