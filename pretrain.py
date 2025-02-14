import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.xLSTM import myxLSTM
import dataset.mit_bih as mit_bih
import dataset.code_15 as code_15
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from trainer import PretrainedxLSTMNetwork

# argparse
import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--lr', type=float, default=0.0008, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.0006, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
parser.add_argument('--embedding_size', type=int, default=64, help='Embedding size')
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
parser.add_argument('--activation_fn', type=str, default='leakyrelu', help='Activation function')
parser.add_argument('--xLSTM_depth', type=int, default=3, help='xLSTM depth')
parser.add_argument('--wandb_log', action='store_true', help='Log to wandb')
parser.add_argument('--normalize', action='store_true', help='Normalize the data')
parser.add_argument('--oversample', action='store_true', help='Oversample the data for the training set')
parser.add_argument('--random_shift', action='store_true', help='Random shift the data on the training set')
parser.add_argument('--data_folder_mit', type=str, default='/media/Volume/data/MIT-BHI/data/', help='Data folder for MIT-BHI dataset')
parser.add_argument('--pretrain_with_code15', action='store_true', help='Pretrain with code15 dataset')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')
parser.add_argument('--nk_clean', action='store_true', help='Use nk_clean for the code15 dataset')
parser.add_argument('--use_scheduler', action='store_true', help='Use the scheduler for the optimizer')
parser.add_argument('--data_folder_code15', type=str, default='/media/Volume/data/CODE15/unlabeled_records_360', help='Data folder for code15 dataset')


def pretrain(config, run=None, wandb=False):

    if not config.pretrain_with_code15:
        dataset = mit_bih.ECGMITBIHDataset(config.data_folder_mit, subset='train', num_leads=1, oversample=config.oversample, random_shift=config.random_shift, patch_size=config.patch_size, normalize=config.normalize, nkclean=config.nk_clean)

        # in this way I should use data that is not present in the training set due to overlapping
        train_dataset, val_dataset = dataset.split_validation_training(val_size=0.1)
        train_dataloader = utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=mit_bih.collate_fn, num_workers=config.num_workers)
        val_dataloader = utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=mit_bih.collate_fn, num_workers=config.num_workers)

    else:
        # full code 15 for training
        train_dataset = code_15.ECGCODE15Dataset(config.data_folder_code15, num_leads=1, random_shift=config.random_shift, patch_size=config.patch_size, normalize=config.normalize)
        # train on mit-bih for validation
        val_dataset = mit_bih.ECGMITBIHDataset(config.data_folder_mit, subset='train', num_leads=1, oversample=False, random_shift=False, patch_size=config.patch_size, normalize=config.normalize, nkclean=config.nk_clean)

        train_dataloader = utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=code_15.collate_fn)
        val_dataloader = utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=mit_bih.collate_fn)

    test_dataset = mit_bih.ECGMITBIHDataset(config.data_folder_mit, subset='test', num_leads=1, oversample=False, random_shift=False, patch_size=config.patch_size, normalize=config.normalize, nkclean=config.nk_clean)
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=mit_bih.collate_fn, num_workers=config.num_workers)
    
    xlstm = myxLSTM(patch_size=config.patch_size, dropout=config.dropout, embedding_dim=config.embedding_size, activation_fn=config.activation_fn, xlstm_depth=config.xLSTM_depth)
    model = PretrainedxLSTMNetwork(model=xlstm, lr=config.lr,optimizer=config.optimizer, batch_size=config.batch_size, wd=config.wd, use_scheduler=config.use_scheduler, patch_size=config.patch_size)
        
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    # early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    if wandb:
        wand_logger = WandbLogger(log_model="all", project="pretrain-xLSTM", experiment=run)
        trainer = L.Trainer(max_epochs=config.epochs, logger=wand_logger, callbacks=[checkpoint_callback]) # , early_stopping])
    else:
        trainer = L.Trainer(max_epochs=config.epochs, callbacks=[checkpoint_callback]) #, early_stopping])

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)

# if main
if __name__ == '__main__':
    args = parser.parse_args()
    pretrain(args, wandb=args.wandb_log)