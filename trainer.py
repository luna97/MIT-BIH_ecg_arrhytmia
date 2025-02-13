from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification

# define the LightningModule
class PretrainedxLSTMNetwork(L.LightningModule):
    def __init__(self, model, lr=1e-3, batch_size=32, optimizer='adam'):
        super().__init__()
        self.lr = lr
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
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
        reconstruct = self.model.reconstruct(x)
        shift_reconstruct = reconstruct[:, :-1]
        x, _, _ = self.model.seq_to_token(x)
        shift_x = x[:, 1:].squeeze()
        # calculate the loss
        mse = nn.functional.mse_loss(shift_reconstruct, shift_x)
        mae = nn.functional.l1_loss(shift_reconstruct, shift_x)
        return mse, mae

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {'optimizer': optimizer } # , 'lr_scheduler': scheduler }
    

class TrainingxLSTMNetwork(L.LightningModule):
    def __init__(self, model, lr=1e-3, batch_size=32, optimizer='adam', num_classes=5):
        super().__init__()
        self.lr = lr
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.save_hyperparameters()
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes)
        self.valid_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes)

    def training_step(self, batch, _):
        loss, preds = self.predict_batch(batch)
        self.train_acc(preds, batch['label'])
        self.train_f1(preds, batch['label'])
        self.log('train_loss', loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, _):
        loss, preds = self.predict_batch(batch)
        self.valid_acc(preds, batch['label'])
        self.valid_f1(preds, batch['label'])
        self.log('val_loss', loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch, _):
        loss, preds = self.reconstruction_step(batch)
        self.test_acc(preds, batch['label'])
        self.test_f1(preds, batch['label'])
        self.log("test_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def predict_batch(self, batch):
        ctx = batch["signal"]
        x = batch['heartbeat']
        targets = batch['label']

        preds = self.model(x)
        loss = nn.functional.cross_entropy(preds, targets)
        return loss, preds
        
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {'optimizer': optimizer } # , 'lr_scheduler': scheduler }
    