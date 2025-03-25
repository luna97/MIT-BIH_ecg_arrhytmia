import os
from torch import utils
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from models.xLSTM import myxLSTM
import dataset.mit_bih as mit_bih
import dataset.code_15 as code_15
import dataset.mimic_iv as mimic
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from trainers.ssl_pretrainer import PretrainedxLSTMNetwork
import sys
import torch

# for debug:
# python3 pretrain.py --epochs 100 --dropout 0.2 --activation_fn relu --batch_size 64 --patch_size 64 --embedding_size 128 --use_scheduler --lr 0.0001 --wd 0.01 --deterministic --xlstm_config m --loss_type mse_grad_min_max --num_workers 32 --nk_clean --pretrain_datasets code15 --random_shift --leads I II III aVR aVL aVF V1 V2 V3 V4 V5 V6 --normalize --random_drop_leads 0.2

# argparse
import argparse
parser = argparse.ArgumentParser(description='Train a model')

# training hyperparameters
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for the dataloader')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
parser.add_argument('--embedding_size', type=int, default=64, help='Embedding size')
parser.add_argument('--deterministic', action='store_true', help='Deterministic training')
parser.add_argument('--use_tab_data', action='store_true', help='Use tabular data')
parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping')
parser.add_argument('--is_sweep', action='store_true', help='Is a sweep')
parser.add_argument('--grad_clip', type=float, default=0.5, help='Gradient clipping value')

# losses params
parser.add_argument('--loss_type', type=str, default='mse', help='Loss type, mse, mae, grad, min_max or a composition of them')
parser.add_argument('--grad_loss_lambda', type=float, default=1., help='Lambda for the contrastive loss')
parser.add_argument('--min_max_loss_lambda', type=float, default=1., help='Lambda for the contrastive loss')
parser.add_argument('--ccc_loss_lambda', type=float, default=1., help='Lambda for the correlation loss')
parser.add_argument('--auto_correlation_loss_lambda', type=float, default=1., help='Lambda for the masked mae loss')

# optimize and scheduler
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
parser.add_argument('--use_scheduler', action='store_true', help='Use the scheduler for the optimizer')
parser.add_argument('--num_epochs_warmup',  type=int, default=2, help='Number of warmup epoch for the scheduler')
parser.add_argument('--num_epochs_warm_restart',  type=int, default=10, help='Number of epoch before restarting the scheduler')
parser.add_argument('--sched_decay_factor', type=float, default=0.8, help='Decay factor for the scheduler')

# model hyperparameters
parser.add_argument('--activation_fn', type=str, default='leakyrelu', help='Activation function')
parser.add_argument('--xlstm_config', type=str, nargs='*', default=['m', 's', 'm', 'm', 'm', 'm', 'm'])
parser.add_argument('--wandb_log', action='store_true', help='Log to wandb')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for the mLSTM module')
parser.add_argument('--xlstm_type', type=str, default='original', help='Type of xLSTM to use')
parser.add_argument('--weight_tying', action='store_true', help='Weight tying')
parser.add_argument('--patch_embedding', default='linear', help='Patch embedding type')
parser.add_argument('--reconstruct_embedding', default='linear', help='Reconstruction head type')
parser.add_argument('--bidirectional', action='store_true', help='Bidirectional LSTM')


# data and augmentations hyperparameters
parser.add_argument('--normalize', action='store_true', help='Normalize the data')
parser.add_argument('--oversample', action='store_true', help='Oversample the data for the training set')
parser.add_argument('--random_shift', action='store_true', help='Random shift the data on the training set')
parser.add_argument('--random_drop_leads', type=float, default=0, help='Randomly drop leads')
parser.add_argument('--random_surrogate_prob', type=float, default=0, help='Random noise')
parser.add_argument('--random_jitter_prob', type=float, default=0, help='Random jitter')
parser.add_argument('--pretrain_datasets', type=str, nargs='*', default=['mimic', 'code15'], help='Datasets to use for pretraining')
parser.add_argument('--nk_clean', action='store_true', help='Use nk_clean for the code15 dataset')
parser.add_argument('--leads', type=str, nargs='*', default=['II'], help='Leads to use for the dataset')
parser.add_argument('--debug', action='store_true', help='Debug mode, uses only a fraction of data for epoch')

# dataset folders
parser.add_argument('--data_folder_mit', type=str, default='/media/Volume/data/MIT-BHI/data', help='Data folder for MIT-BHI dataset')
parser.add_argument('--data_folder_code15', type=str, default='/media/Volume/data/CODE15/nkclean_360_12l', help='Data folder for code15 dataset')
parser.add_argument('--labels_file_code15', type=str, default='/media/Volume/data/CODE15/exams.csv', help='Labels file for code15 dataset')
parser.add_argument('--data_folder_mimic', type=str, default='/media/Volume/data/MIMIC_IV/nkclean_360_12l', help='Data folder for MIMIC dataset')
parser.add_argument('--labels_file_mimic', type=str, default='/media/Volume/data/MIMIC_IV/records_w_diag_icd10_labelled.csv', help='Labels file for MIMIC dataset')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to save the checkpoints')

def pretrain(config, run=None, wandb=False):
    max_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", config.num_workers))
    config.num_workers = min(config.num_workers, max_cpus)

    # set deterministic training
    if config.deterministic: L.seed_everything(42)

    datasets_pretrain = []

    for dataset in config.pretrain_datasets:
        if dataset == 'mimic':
            datasets_pretrain.append(mimic.ECGMIMICDataset(config, leads_to_use=config.leads))
        elif dataset == 'code15':
            datasets_pretrain.append(code_15.ECGCODE15Dataset(config, leads_to_use=config.leads))
        else:
            raise ValueError(f"Dataset {dataset} not found")

    val_dataset = mit_bih.ECGMITBIHDataset(config, subset='train', random_shift=False)

    train_dataset = utils.data.ConcatDataset(datasets_pretrain)
    # keep only 10% of the dataset
    if config.debug: train_dataset = utils.data.Subset(train_dataset, range(0, len(train_dataset) // 100))
    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=code_15.collate_fn)

    # cat the two dataloaders
    if config.debug: val_dataset = utils.data.Subset(val_dataset, range(0, len(val_dataset) // 10))
    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=mit_bih.collate_fn)

    len_train_dataset = len(train_dataset)
    test_dataset = mit_bih.ECGMITBIHDataset(config, subset='test', random_shift=False)
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=mit_bih.collate_fn, num_workers=config.num_workers)
    
    xlstm = myxLSTM(config=config, num_classes=5, num_channels=len(config.leads))
    if config.checkpoint != None:
        model = PretrainedxLSTMNetwork.load_from_checkpoint(checkpoint_path=config.checkpointg)
    else:
        model = PretrainedxLSTMNetwork(model=xlstm, len_train_dataset=len_train_dataset, config=config)
        
    checkpoint_callback = ModelCheckpoint(monitor='val_nrmse')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    early_stopping = EarlyStopping(monitor='val_nrmse', patience=config.patience)

    if wandb:
        wand_logger = WandbLogger(project="pretrain-xLSTM", experiment=run)
        wand_logger.watch(model, log='gradients')
        trainer = L.Trainer(max_epochs=config.epochs, logger=wand_logger, callbacks=[checkpoint_callback, lr_monitor, early_stopping], gradient_clip_val=config.grad_clip)
    else:
        trainer = L.Trainer(max_epochs=config.epochs, callbacks=[checkpoint_callback, lr_monitor, early_stopping], gradient_clip_val=config.grad_clip)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = parser.parse_args()
    pretrain(args, wandb=args.wandb_log)