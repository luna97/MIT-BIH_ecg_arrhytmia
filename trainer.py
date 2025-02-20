from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
from utils.train_utils import masked_mse_loss, masked_mae_loss, gradient_loss
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# define the LightningModule
class PretrainedxLSTMNetwork(L.LightningModule):
    def __init__(
            self, 
            model, 
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
        self.config = config
        self.save_hyperparameters()

    def training_step(self, batch, _):
        loss = self.reconstruct_batch(batch, step='train')
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, _):
        loss = self.reconstruct_batch(batch)
        self.log("val_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch, _):
        loss = self.reconstruct_batch(batch)
        self.log("test_mse", mse.item(), prog_bar=True, batch_size=self.batch_size)
        return loss

    
    def reconstruct_batch(self, batch, step):
        x = batch["signal"]
        mask = batch["mask"]
        tab_data = batch["tab_data"]

        if not self.multi_token_prediction:

            reconstruct = self.model.reconstruct(x, tab_data)

            # get rid of the exeding part on the original signal
            shift_x = x[:, :reconstruct.shape[1]]
            shift_x = shift_x[:, self.patch_size:].squeeze()

            mask_shifted = mask[:, :reconstruct.shape[1]]
            mask_shifted = mask_shifted[:, self.patch_size:].squeeze()

            shift_reconstruct = reconstruct[:, :-self.patch_size]

            # calculate the loss
            mae = masked_mae_loss(shift_reconstruct, shift_x, mask=mask_shifted)
            mse = masked_mse_loss(shift_reconstruct, shift_x, mask=mask_shifted)
            grad_loss = gradient_loss(shift_reconstruct, shift_x, mask=mask_shifted)

        else:
            r1, r2, r3, r4 = self.model.reconstruct(x, tab_data)
            # get rid of exeding part on the original signal
            shift_x = x[:, :r1.shape[1]]

            shift_x1 = shift_x[:, self.patch_size:].squeeze()
            shift_x2 = shift_x[:, self.patch_size * 2:].squeeze()
            shift_x3 = shift_x[:, self.patch_size * 3:].squeeze()
            shift_x4 = shift_x[:, self.patch_size * 4:].squeeze()

            mask_shifted = mask[:, :r1.shape[1]]
            mask_shifted1 = mask_shifted[:, self.patch_size:].squeeze()
            mask_shifted2 = mask_shifted[:, self.patch_size * 2:].squeeze()
            mask_shifted3 = mask_shifted[:, self.patch_size * 3:].squeeze()
            mask_shifted4 = mask_shifted[:, self.patch_size * 4:].squeeze()

            shift_reconstruct1 = r1[:, :-self.patch_size]
            shift_reconstruct2 = r2[:, :-self.patch_size * 2]
            shift_reconstruct3 = r3[:, :-self.patch_size * 3]
            shift_reconstruct4 = r4[:, :-self.patch_size * 4]

            mae1 = masked_mae_loss(shift_reconstruct1, shift_x1, mask=mask_shifted1)
            mae2 = masked_mae_loss(shift_reconstruct2, shift_x2, mask=mask_shifted2)
            mae3 = masked_mae_loss(shift_reconstruct3, shift_x3, mask=mask_shifted3)
            mae4 = masked_mae_loss(shift_reconstruct4, shift_x4, mask=mask_shifted4)
            mae = (mae1 + mae2 + mae3 + mae4) / 4

            mse1 = masked_mse_loss(shift_reconstruct1, shift_x1, mask=mask_shifted1)
            mse2 = masked_mse_loss(shift_reconstruct2, shift_x2, mask=mask_shifted2)
            mse3 = masked_mse_loss(shift_reconstruct3, shift_x3, mask=mask_shifted3)
            mse4 = masked_mse_loss(shift_reconstruct4, shift_x4, mask=mask_shifted4)
            mse = (mse1 + mse2 + mse3 + mse4) / 4

            grad_loss1 = gradient_loss(shift_reconstruct1, shift_x1, mask=mask_shifted1)
            grad_loss2 = gradient_loss(shift_reconstruct2, shift_x2, mask=mask_shifted2)
            grad_loss3 = gradient_loss(shift_reconstruct3, shift_x3, mask=mask_shifted3)
            grad_loss4 = gradient_loss(shift_reconstruct4, shift_x4, mask=mask_shifted4)
            grad_loss = (grad_loss1 + grad_loss2 + grad_loss3 + grad_loss4) / 4


        self.log(f"{step}_mse", mse.item(), prog_bar=True, batch_size=self.batch_size)
        self.log(f"{step}_mae", mae.item(), prog_bar=True, batch_size=self.batch_size)
        self.log(f"{step}_grad", grad_loss.item(), prog_bar=True, batch_size=self.batch_size)

        if self.loss_type == 'mae':
            return mae
        elif self.loss_type == 'grad':
            return grad_loss + mse 
        else:
            return mse

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)

        if self.use_scheduler:
            sched = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            scheduler = {
                'scheduler': sched,
                'interval': 'epoch', # or 'step' 
                'frequency': 1,
                'monitor': 'val_loss',
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
    