import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.xLSTM import myxLSTM
from dataset import ECGDataset, collate_fn
import wandb
import yaml
import json
from lightning.pytorch.callbacks import ModelCheckpoint

# argparse
import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
parser.add_argument('--embedding_size', type=int, default=64, help='Embedding size')
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
parser.add_argument('--activation_fn', type=str, default='leakyrelu', help='Activation function')
parser.add_argument('--xLSTM_depth', type=int, default=3, help='xLSTM depth')
parser.add_argument('--wandb_log', action='store_true', help='Log to wandb')
parser.add_argument('--normalize', action='store_true', help='Normalize the data')

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
        loss, mae = self.reconstruction_step(batch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log("train_mae", mae.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, _):
        loss, mae = self.reconstruction_step(batch)
        self.log("val_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log("val_mae", mae.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch, _):
        loss, mae = self.reconstruction_step(batch)
        self.log("test_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        self.log("test_mae", mae.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def reconstruction_step(self, batch):
        x = batch["window_signals"]
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
    

def train(config, run=None, wandb=False):
    patch_size = config.patch_size
    batch_size = config.batch_size
    dropout = config.dropout
    embedding_dim = config.embedding_size
    optimizer = config.optimizer
    epochs = config.epochs
    lr = config.lr
    activation_fn = config.activation_fn
    xLSTM_depth = config.xLSTM_depth

    xlstm = myxLSTM(patch_size=patch_size, dropout=dropout, embedding_dim=embedding_dim, activation_fn=activation_fn, xlstm_depth=xLSTM_depth)
    model = PretrainedxLSTMNetwork(model=xlstm, lr=lr,optimizer=optimizer, batch_size=batch_size)
        
    dataset = ECGDataset('/media/Volume/data/MIT-BHI/data/', subset='train', num_leads=1, oversample=True, random_shift_window=True, patch_size=patch_size, normalize=config.normalize)

    # in this way I should use data that is not present in the training set due to overlapping
    train_dataset, val_dataset = dataset.split_validation_training(val_size=0.1)
    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    test_dataset = ECGDataset('/media/Volume/data/MIT-BHI/data/', subset='test', num_leads=1, oversample=False, random_shift_window=False, patch_size=patch_size, normalize=config.normalize)
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    if wandb:
        wand_logger = WandbLogger(log_model="all", project="pretrain-xLSTM", experiment=run)
        trainer = L.Trainer(max_epochs=epochs, logger=wand_logger, callbacks=[checkpoint_callback])
    else:
        trainer = L.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)

# if main
if __name__ == '__main__':
    args = parser.parse_args()
    train(args, wandb=args.wandb_log)