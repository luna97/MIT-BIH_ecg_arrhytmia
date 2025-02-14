import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.xLSTM import myxLSTM
import dataset.mit_bih as mit_bih
import dataset.code_15 as code_15
from lightning.pytorch.callbacks import ModelCheckpoint
from trainer import TrainingxLSTMNetwork
import torch
from utils.utils import get_training_class_weights

# argparse
import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--lr_head', type=float, default=0.0001, help='Learning rate for the classification head')
parser.add_argument('--lr_xlstm', type=float, default=0.0001, help='Learning rate for the xLSTM')
parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout')
parser.add_argument('--embedding_size', type=int, default=64, help='Embedding size')
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
parser.add_argument('--activation_fn', type=str, default='gelu', help='Activation function')
parser.add_argument('--xLSTM_depth', type=int, default=5, help='xLSTM depth')
parser.add_argument('--wandb_log', action='store_true', help='Log to wandb')
parser.add_argument('--normalize', action='store_true', help='Normalize the data')
parser.add_argument('--oversample', action='store_true', help='Oversample the data for the training set')
parser.add_argument('--random_shift', action='store_true', help='Random shift the data on the training set')
parser.add_argument('--data_folder_mit', type=str, default='/media/Volume/data/MIT-BHI/data/', help='Data folder for MIT-BHI dataset')
parser.add_argument('--checkpoint', type=str, help='Checkpoint name', required=True)


def train(config, run=None, wandb=False):
    dataset = mit_bih.ECGMITBIHDataset(args.data_folder_mit, subset='train', num_leads=1, oversample=False, random_shift=config.random_shift, patch_size=config.patch_size, normalize=config.normalize)
    train_dataset, val_dataset = dataset.split_validation_training(val_size=0.1)

    weights = get_training_class_weights(train_dataset).to('cuda')


    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=mit_bih.collate_fn)
    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=mit_bih.collate_fn)

    test_dataset = mit_bih.ECGMITBIHDataset(config.data_folder_mit, subset='test', num_leads=1, oversample=False, random_shift=False, patch_size=config.patch_size, normalize=config.normalize)
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=mit_bih.collate_fn, num_workers=4)

    checkpoint = torch.load(args.checkpoint)
    new_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}

    xlstm = myxLSTM(patch_size=config.patch_size, dropout=config.dropout, embedding_dim=config.embedding_size, activation_fn=config.activation_fn, xlstm_depth=config.xLSTM_depth)
    xlstm.load_state_dict(new_state_dict)   
    model = TrainingxLSTMNetwork(model=xlstm, lr_head=config.lr_head, lr_xlstm=config.lr_xlstm, optimizer=config.optimizer, batch_size=config.batch_size, wd=config.wd, weights=weights)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    if wandb:
        wand_logger = WandbLogger(log_model="all", project="train-xLSTM", experiment=run)
        trainer = L.Trainer(max_epochs=config.epochs, logger=wand_logger, callbacks=[checkpoint_callback])
    else:
        trainer = L.Trainer(max_epochs=config.epochs, callbacks=[checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)

# if main
if __name__ == '__main__':
    args = parser.parse_args()
    train(args, wandb=args.wandb_log)