import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.xLSTM import myxLSTM
import dataset.mit_bih as mit_bih
import dataset.code_15 as code_15
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from trainer import PretrainedxLSTMNetwork
import sys


# python pretrain.py --epochs 10 --dropout 0.3 --activation_fn relu --batch_size 784 --patch_size 128 --embedding_size 784 --use_scheduler --lr 0.0005 --wd 0.0001 --multi_token_prediction --data_folder_code15 /media/Volume/data/CODE15/unlabeled_records_360 --use_scheduler --series_decomposition 0 --deterministic --xlstm_config m s m --loss_type grad --pretrain_with_code15 --num_workers 32

# argparse
import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
parser.add_argument('--embedding_size', type=int, default=64, help='Embedding size')
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
parser.add_argument('--activation_fn', type=str, default='leakyrelu', help='Activation function')
parser.add_argument('--xlstm_config', type=str, nargs='*', default=['m', 's', 'm', 'm', 'm', 'm', 'm'])
parser.add_argument('--wandb_log', action='store_true', help='Log to wandb')
parser.add_argument('--normalize', action='store_true', help='Normalize the data')
parser.add_argument('--oversample', action='store_true', help='Oversample the data for the training set')
parser.add_argument('--random_shift', action='store_true', help='Random shift the data on the training set')
parser.add_argument('--data_folder_mit', type=str, default='/media/Volume/data/MIT-BHI/data/', help='Data folder for MIT-BHI dataset')
parser.add_argument('--pretrain_with_code15', action='store_true', help='Pretrain with code15 dataset')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for the dataloader')
parser.add_argument('--nk_clean', action='store_true', help='Use nk_clean for the code15 dataset')
parser.add_argument('--use_scheduler', action='store_true', help='Use the scheduler for the optimizer')
parser.add_argument('--data_folder_code15', type=str, default='/media/Volume/data/CODE15/unlabeled_records_360', help='Data folder for code15 dataset')
parser.add_argument('--patch_embedding', type=str, default='linear', help='Patch embedding type')
parser.add_argument('--multi_token_prediction', action='store_true', help='Multi token prediction')
parser.add_argument('--loss_type', type=str, default='mse', help='Loss type')
parser.add_argument('--num_epochs_warmup',  type=int, default=2, help='Number of warmup epoch for the scheduler')
parser.add_argument('--num_epochs_warm_restart',  type=int, default=5, help='Number of epoch before restarting the scheduler')
parser.add_argument('--deterministic', action='store_true', help='Deterministic training')
parser.add_argument('--is_sweep', action='store_true', help='Is a sweep')
parser.add_argument('--series_decomposition', type=int, default=0, help='Series decomposition kernel size')
parser.add_argument('--grad_clip', type=float, default=5, help='Gradient clipping value')
parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping')
parser.add_argument('--sched_decay_factor', type=float, default=0.8, help='Decay factor for the scheduler')
parser.add_argument('--use_tab_data', action='store_true', help='Use tabular data')

def pretrain(config, run=None, wandb=False):

    if config.deterministic:
        L.seed_everything(42)

    if not config.pretrain_with_code15:
        dataset = mit_bih.ECGMITBIHDataset(config, subset='train')

        # in this way I should use data that is not present in the training set due to overlapping
        train_dataset, val_dataset = dataset.split_validation_training(val_size=0.1)
        train_dataloader = utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=mit_bih.collate_fn, num_workers=config.num_workers)
        val_dataloader = utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=mit_bih.collate_fn, num_workers=config.num_workers)

    else:
        # full code 15 for training
        train_dataset = code_15.ECGCODE15Dataset(config)
        # train on mit-bih for validation
        val_dataset = mit_bih.ECGMITBIHDataset(config, subset='train')

        train_dataloader = utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=code_15.collate_fn)
        val_dataloader = utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=mit_bih.collate_fn)

    len_train_dataset = len(train_dataset)
    test_dataset = mit_bih.ECGMITBIHDataset(config, subset='test')
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=mit_bih.collate_fn, num_workers=config.num_workers)
    
    
    xlstm = myxLSTM(config=config, num_classes=5)
    model = PretrainedxLSTMNetwork(model=xlstm, len_train_dataset=len_train_dataset, config=config)
        
    checkpoint_callback = ModelCheckpoint(monitor='val_nrmse')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    early_stopping = EarlyStopping(monitor='val_nrmse', patience=config.patience)

    if wandb:
        wand_logger = WandbLogger(project="pretrain-xLSTM", experiment=run)
        trainer = L.Trainer(max_epochs=config.epochs, logger=wand_logger, callbacks=[checkpoint_callback, lr_monitor, early_stopping], gradient_clip_val=config.grad_clip)
    else:
        trainer = L.Trainer(max_epochs=config.epochs, callbacks=[checkpoint_callback, lr_monitor, early_stopping], gradient_clip_val=config.grad_clip)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)

# if main
if __name__ == '__main__':
    args = parser.parse_args()
    pretrain(args, wandb=args.wandb_log)