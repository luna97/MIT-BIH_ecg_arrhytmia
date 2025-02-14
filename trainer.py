from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification

# define the LightningModule
class PretrainedxLSTMNetwork(L.LightningModule):
    def __init__(
            self, 
            model, 
            lr=1e-3, 
            batch_size=32, 
            optimizer='adam', 
            wd=0.0001,
            use_scheduler=False,
            scheduler_factor=0.1,
            scheduler_patience=5,
            patch_size=64
        ):
        super().__init__()
        self.lr = lr
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.wd = wd
        self.use_scheduler = use_scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.patch_size = patch_size
        self.save_hyperparameters()

    def training_step(self, batch, _):
        loss, mae = self.reconstruct_batch(batch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log("train_mae", mae.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, _):
        loss, mae = self.reconstruct_batch(batch)
        self.log("val_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log("val_mae", mae.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch, _):
        loss, mae = self.reconstruct_batch(batch)
        self.log("test_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log("test_mae", mae.item(), prog_bar=True, batch_size=self.batch_size)
        return loss

    
    def reconstruct_batch(self, batch):
        x = batch["signal"]
        mask = batch["mask"]
        # print('initial mask shape', mask.shape)
        # print('initial x shape', x.shape)
        reconstruct = self.model.reconstruct(x)
        # print('reconstruct shape', reconstruct.shape)
        # shift_reconstruct = reconstruct[:, :-1]
        # shift_reconstruct = shift_reconstruct

        # get rid of the exeding part on the original signal
        shift_x = x[:, :reconstruct.shape[1]]
        shift_x = shift_x[:, self.patch_size:].squeeze()

        mask_shifted = mask[:, :reconstruct.shape[1]]
        mask_shifted = mask_shifted[:, self.patch_size:].squeeze()

        shift_reconstruct = reconstruct[:, :-self.patch_size]

        #print('mask shape', mask.shape)
        #print('x shape', x.shape)   
        #print('reconstruct shape', shift_reconstruct.shape)

        shift_reconstruct = shift_reconstruct.masked_fill(mask_shifted, 0)
        shift_x = shift_x.masked_fill(mask_shifted, 0)

        # calculate the loss
        mse = nn.functional.mse_loss(shift_reconstruct, shift_x)
        mae = nn.functional.l1_loss(shift_reconstruct, shift_x)
        return mse, mae

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)

        if self.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                verbose=True
            )
            scheduler = {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Metric to monitor
                'interval': 'epoch',
                'frequency': 1  
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
    