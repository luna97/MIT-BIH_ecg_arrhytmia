from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
from utils.train_utils import masked_mse_loss, masked_mae_loss, gradient_loss, masked_min_max_loss
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
from utils.loss_utils import contrastive_coupled_loss


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
        self.contrastive_loss_lambda = config.contrastive_loss_lambda
        self.epochs = config.epochs
        self.train_acc = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=num_classes, top_k=1, average='micro')
        self.valid_acc = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=num_classes, top_k=1, average='micro')
        self.test_acc = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=num_classes, top_k=1, average='micro')
        self.test_acc_no_avg = torchmetrics.classification.accuracy.MulticlassAccuracy(num_classes=num_classes, top_k=1, average=None)
        self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, top_k=1, average='macro')
        self.valid_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, top_k=1, average='macro')
        self.test_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, top_k=1, average=None)
        # add sensitivity and specificity for the first class
        self.val_spec = torchmetrics.classification.specificity.MulticlassSpecificity(num_classes=num_classes, average=None)
        self.test_spec = torchmetrics.classification.specificity.MulticlassSpecificity(num_classes=num_classes, average=None)
        self.val_recall = torchmetrics.classification.precision_recall.MulticlassRecall(num_classes=num_classes, average=None)
        self.test_recall = torchmetrics.classification.precision_recall.MulticlassRecall(num_classes=num_classes, average=None)
        self.val_precision = torchmetrics.classification.precision_recall.MulticlassPrecision(num_classes=num_classes, average=None)
        self.test_precision = torchmetrics.classification.precision_recall.MulticlassPrecision(num_classes=num_classes, average=None)
        if not config.is_sweep:
            self.save_hyperparameters()

    def training_step(self, batch, _):
        loss, contrastive_loss, _, preds = self.predict_batch(batch)
        self.train_acc(preds, batch['label'])
        self.train_f1(preds, batch['label'])
        self.log('train_loss', loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log('train_acc', self.train_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('train_f1', self.train_f1, prog_bar=True, batch_size=self.batch_size)

        if self.contrastive_loss_lambda > 0:
            self.log('train_contrastive_loss', contrastive_loss.item(), prog_bar=True, batch_size=self.batch_size)
            return loss + contrastive_loss * self.contrastive_loss_lambda
        else:
            return loss
    
    def validation_step(self, batch, _):
        loss, contrastive_loss, _, preds = self.predict_batch(batch)
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

        if self.contrastive_loss_lambda > 0:
            self.log('val_contrastive_loss', contrastive_loss.item(), prog_bar=True, batch_size=self.batch_size)
            return loss + contrastive_loss * self.contrastive_loss_lambda
        else:
            return loss
            
    def test_step(self, batch, _):
        loss, contrastive_loss, _, preds = self.predict_batch(batch)

        self.test_acc = self.test_acc.to(preds.device)
        self.test_acc(preds, batch['label'])

        self.test_acc_no_avg = self.test_acc_no_avg.to(preds.device)
        self.test_acc_no_avg(preds, batch['label'])

        self.test_f1 = self.test_f1.to(preds.device)
        self.test_f1(preds, batch['label'])

        self.log("test_loss", loss.item(), batch_size=self.batch_size)
        self.log("test_acc", self.test_acc, batch_size=self.batch_size)

        #accuracy
        self.log("test_acc/N", self.test_acc_no_avg[0], batch_size=self.batch_size, metric_attribute='test_acc')
        self.log("test_acc/S", self.test_acc_no_avg[1], batch_size=self.batch_size, metric_attribute='test_acc')
        self.log("test_acc/V", self.test_acc_no_avg[2], batch_size=self.batch_size, metric_attribute='test_acc')
        self.log("test_acc/F", self.test_acc_no_avg[3], batch_size=self.batch_size, metric_attribute='test_acc')
        self.log("test_acc/Q", self.test_acc_no_avg[4], batch_size=self.batch_size, metric_attribute='test_acc')

        # f1
        self.log("test_f1/N", self.test_f1[0], batch_size=self.batch_size, metric_attribute='test_f1')
        self.log("test_f1/S", self.test_f1[1], batch_size=self.batch_size, metric_attribute='test_f1')
        self.log("test_f1/V", self.test_f1[2], batch_size=self.batch_size, metric_attribute='test_f1')
        self.log("test_f1/F", self.test_f1[3], batch_size=self.batch_size, metric_attribute='test_f1')
        self.log("test_f1/Q", self.test_f1[4], batch_size=self.batch_size, metric_attribute='test_f1')
        self.log("test_f1/mean", (self.test_f1[0] + self.test_f1[1] + self.test_f1[2] + self.test_f1[3] + self.test_f1[4]) / 5, batch_size=self.batch_size, metric_attribute='test_f1')

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

        if self.contrastive_loss_lambda > 0:
            # self.log("test_contrastive_loss", contrastive_loss.item(), batch_size=self.batch_size)
            return loss + contrastive_loss * self.contrastive_loss_lambda
        else:
            return loss
            
    
    def predict_batch(self, batch):
        ctx = batch["signal"]
        x = batch['heartbeat']
        tab_data = batch['tab_data'] if 'tab_data' in batch.keys() else None 
        targets = batch['label']
        out, cls_token = self.model(ctx, x, tab_data)
        preds = torch.argmax(out, dim=1)
        loss = nn.functional.cross_entropy(out, targets, weight=self.weights)
        if self.contrastive_loss_lambda > 0:
            contrastive_loss = contrastive_coupled_loss(cls_token, targets, batch['patient_ids'], class_weights=self.weights) * 0.1
            return loss, contrastive_loss, out, preds
        else:
            return loss, 0, out, preds

    def get_params(self):
        return [
            # head and sep token with normal lr
            {'params': self.model.fc.parameters(), 'lr': self.lr_head, 'weight_decay': self.wd},
            {'params': self.model.sep_token, 'lr': self.lr_head, 'weight_decay': self.wd},
            {'params': self.model.cls_token, 'lr': self.lr_head, 'weight_decay': self.wd},

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
