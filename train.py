import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.xLSTM import myxLSTM
import dataset.mit_bih as mit_bih
import dataset.code_15 as code_15
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from trainers.mit_bih_trainer import TrainingxLSTMNetwork
import torch
from utils.utils import get_training_class_weights
import argparse

# training hyperparameters
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--lr_head', type=float, default=0.0001, help='Learning rate for the classification head')
parser.add_argument('--lr_xlstm', type=float, default=0.0001, help='Learning rate for the xLSTM')
parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for the dataloader')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--patch_size', type=int, default=128, help='Patch size')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout')
parser.add_argument('--embedding_size', type=int, default=64, help='Embedding size')
parser.add_argument('--deterministic', action='store_true', help='Deterministic training')
parser.add_argument('--use_tab_data', action='store_true', help='Use tabular data')
parser.add_argument('--patience', type=int, default=5, help='Patience for the early stopping')
parser.add_argument('--is_sweep', action='store_true', help='Is a sweep')
parser.add_argument('--grad_clip', type=float, default=5, help='Gradient clipping value')
parser.add_argument('--weight_tying', action='store_true', help='Weight tying')
parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for the loss function')

# optimize and scheduler
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
parser.add_argument('--use_scheduler', action='store_true', help='Use the scheduler for the optimizer')
parser.add_argument('--num_epochs_warmup',  type=int, default=1, help='Number of warmup epoch for the scheduler')
parser.add_argument('--num_epochs_warm_restart',  type=int, default=3, help='Number of epoch before restarting the scheduler')
parser.add_argument('--sched_decay_factor', type=float, default=0.8, help='Decay factor for the scheduler')
parser.add_argument('--contrastive_loss_lambda', type=float, default=0., help='Lambda for the contrastive loss')
parser.add_argument('--random_surrogate_prob', type=float, default=0., help='Probability of using a surrogate')
parser.add_argument('--random_jitter_prob', type=float, default=0., help='Probability of using jitter')
parser.add_argument('--loss_type', type=str, default='')
parser.add_argument('--split_by_patient', action='store_true', help='Split the dataset in val and train by patient')

# model hyperparameters
parser.add_argument('--activation_fn', type=str, default='relu', help='Activation function')
parser.add_argument('--xlstm_config', type=str, nargs='*', default=['m', 's', 'm', 'm', 'm', 'm', 'm'])
parser.add_argument('--wandb_log', action='store_true', help='Log to wandb')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for the mLSTM module')

# data and augmentations hyperparameters
parser.add_argument('--normalize', action='store_true', help='Normalize the data')
parser.add_argument('--oversample', action='store_true', help='Oversample the data for the training set')
parser.add_argument('--random_shift', action='store_true', help='Random shift the data on the training set')
parser.add_argument('--random_drop_leads', type=float, default=0, help='Randomly drop leads')
parser.add_argument('--nk_clean', action='store_true', help='Use nk_clean for the code15 dataset')
parser.add_argument('--leads', type=str, nargs='*', default=['II'], help='Leads to use for the dataset')

# data folders and paths
parser.add_argument('--data_folder_mit', type=str, default='/media/Volume/data/MIT-BHI/data/', help='Data folder for MIT-BHI dataset')
parser.add_argument('--checkpoint', type=str, help='Checkpoint name')


def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)

    dataset =  mit_bih.ECGMITBIHDataset(config, subset='train', use_labels_in_tab_data=False, random_shift=config.random_shift)
    train_dataset, val_dataset = dataset.split_validation_training(val_size=0.1, split_by_patient=config.split_by_patient)

    if config.use_class_weights:
        # weights = get_training_class_weights(train_dataset).to('cuda')
        # real_weights = [2.2248e-01, 1.0810e+01, 2.6938e+00, 2.4588e+01, 1.2755e+03]
        # without the last class = [0.2781, 13.5098,  3.3668, 30.7307]
        weights = torch.tensor([0.2781, 13.5098,  3.3668, 30.7307, 1]).to('cuda')
    else:
        weights = None

    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=mit_bih.collate_fn)
    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=mit_bih.collate_fn)

    test_dataset = mit_bih.ECGMITBIHDataset(config, subset='test', use_labels_in_tab_data=False, random_shift=config.random_shift)
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=mit_bih.collate_fn, num_workers=config.num_workers)

    xlstm = myxLSTM(config=config, num_classes=5, num_channels=len(config.leads))

    if config.checkpoint is not None and config.checkpoint != '':   
        checkpoint = torch.load(config.checkpoint)
        new_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        message = xlstm.load_state_dict(new_state_dict, strict=False) 
        print(message) 

    model = TrainingxLSTMNetwork(model=xlstm, config=config, len_train_dataset=len(train_dataset), num_classes=5, weights=weights)

    # checkpoint_callback = ModelCheckpoint(monitor='val_f1', mode='max')
    early_stopping = EarlyStopping(monitor='val_f1', patience=config.patience, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')


    if wandb:
        wand_logger = WandbLogger(project="train-xLSTM", experiment=run)
        wand_logger.watch(model, log='gradients')
        trainer = L.Trainer(max_epochs=config.epochs, logger=wand_logger, callbacks=[early_stopping, lr_monitor], gradient_clip_val=config.grad_clip)
    else:
        trainer = L.Trainer(max_epochs=config.epochs, callbacks=[early_stopping, lr_monitor], gradient_clip_val=config.grad_clip)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)

# if main
if __name__ == '__main__':
    args = parser.parse_args()
    train(args, wandb=args.wandb_log)
    