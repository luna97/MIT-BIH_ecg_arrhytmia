import torch
import warnings
warnings.filterwarnings('always')
from old.dataset_old import ECGDataset, collate_fn
from models.simple_LSTM import ECG_LSTM, ECG_CONV1D_LSTM
from models.xLSTM import myxLSTM
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import Counter
from augmentations import *
from torchvision import transforms
import argparse
import random
import wandb
import os 
from utils.utils import *
from utils.train_utils import *
from utils.loss_utils import *

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from utils.utils import calculate_metrics, print_metrics_table

# Argument parser for hyperparameters
parser = argparse.ArgumentParser(description='Train ECG model')
parser.add_argument('--input_size', type=int, default=2, help='Input size')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate for training')
parser.add_argument('--lr_pretrain', type=float, default=0.0003, help='Learning rate for pretraining')
parser.add_argument('--wd', type=float, default=0.00001, help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--epochs_pretrain', type=int, default=50, help='Number of epochs for pretraining')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--augment_data', action='store_true', help='Augment data')
parser.add_argument('--model', type=str, default='xLSTM', help='Model to use')
parser.add_argument('--skip_validation', action='store_true', help='Skip validation')
parser.add_argument('--dataset', type=str, default='base', help='Dataset to use')
parser.add_argument('--deterministic', action='store_true', help='Deterministic', default=False)
parser.add_argument('--log', action='store_true', help='Log to wandb')
parser.add_argument('--run_name', type=str, default=None, help='Run name')
parser.add_argument('--use_scheduler', action='store_true', help='Use scheduler')
parser.add_argument('--pooling', type=str, default='avg', help='Pooling method')
parser.add_argument('--act_fn', type=str, default='leakyrelu', help='Activation function')
parser.add_argument('--classes', nargs='*', type=str, default=['N', 'S', 'V'], help='Classes')
parser.add_argument('--num_leads', type=int, default=2, help='Number of leads')
parser.add_argument('--include_val', action='store_true', help='Include validation set', default=False)
parser.add_argument('--label_smoothing', type=float, default=.0, help='Label smoothing')
parser.add_argument('--pretrain', action='store_true', help='Pretrain model')
parser.add_argument('--sparse_loss', action='store_true', help='Use sparse loss')
parser.add_argument('--cluster_loss', action='store_true', help='Use cluster loss')
parser.add_argument('--separate_pretrain', action='store_true', help='Separate pretrain')
parser.add_argument('--data_dir', type=str, default='/media/Volume/data/MIT-BHI/data', help='Data directory')
parser.add_argument('--linear_probing', action='store_true', help='Linear probing')
args = parser.parse_args()


# setup deterministic seeds
if args.deterministic:
    # setup the seeds and return the generator to use for the data loader
    generator = setup_determinitic_seeds()

# define the transform
if not args.augment_data:
    transform = None
else: 
    transform = transforms.Compose([
        CropResizing(start_idx=0, resize=False),
        FTSurrogate(phase_noise_magnitude=0.075, prob=0.5),
        Jitter(sigma=0.2),
        Rescaling(sigma=0.5)
    ])

# get the datasets
train_dataset, val_dataset, test_dataset = get_datasets(args, transform, args.data_dir)

# get the dataloaders
if args.deterministic:
    # setting the seed for the worker and the generator
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=generator)
else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = len(args.classes)
weights = get_training_class_weights(train_dataset, num_classes).to(device)

model = get_model(args)

if args.linear_probing:
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)  # Use appropriate loss for multi-class classification

# Add Cosine Annealing LR Scheduler
if args.use_scheduler:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # T_max is the number of iterations for the first half of the cycle


if args.log:
    project_name = f"ecg-classification-{len(args.classes)}class"
    wandb.init(project=project_name, config=args, name=args.run_name)
    wandb.watch(model)
    # log model architecture to wandb
    wandb.config.update({"model_architecture": str(model)})


# 3. Training Loop
model.to(device)

best_f1, best_model_state = 0.0, None

# Pretrain the model
if args.pretrain:
    best_val_loss = float('inf')
    optimizer_encoder = optim.AdamW([
        param for name, param in model.named_parameters() if 'fc' not in name
    ], lr=args.lr_pretrain, weight_decay=args.wd)  # Use appropriate optimizer

    if args.use_scheduler:
        scheduler_encoder = optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=args.epochs_pretrain)

    if not args.separate_pretrain:
        # if no separate pretrain, set the epochs to the same value
        args.epochs_pretrain = args.epochs

    for epoch in range(args.epochs_pretrain):
        res_pretrain_loop = pretrain_loop(model, optimizer_encoder, criterion, train_loader, device, args, epoch, weights)

        if not args.separate_pretrain:
            if args.use_scheduler: scheduler.step()

        if args.skip_validation: continue

        pretrain_loss_val = eval_pretrain_loop(model, val_loader, device, weights, args)

        if args.separate_pretrain:
            ## evaluation for pretraining only
            if args.use_scheduler: scheduler_encoder.step()

            if args.log: wandb.log({
                "pretrain_loss": res_pretrain_loop['pretrain_loss'],
                "pretrain_val_loss": pretrain_loss_val
            })
            # keep track of the best model based on validation loss
            if pretrain_loss_val < best_val_loss:
                best_val_loss = pretrain_loss_val
                best_model_state = model.state_dict()
        else:
            # evaluation loop for training head
            val_loss, val_accuracy, val_f1, _, _, _ = eval_loop(model, criterion, val_loader, device, num_classes)
            print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

            # Save the best model based on validation F1 score
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = model.state_dict()

            # Log epoch metrics to wandb
            if args.log: wandb.log({
                "train_loss": res_pretrain_loop['train_loss'],
                "train_accuracy": res_pretrain_loop['accuracy_train'],
                "train_f1":res_pretrain_loop['f1_train'],
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
                "pretrain_loss": res_pretrain_loop['pretrain_loss'],
                "pretrain_val_loss": pretrain_loss_val
            })

    # Load the best model state if pretrained
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

if args.separate_pretrain or not args.pretrain:
    for epoch in range(args.epochs):
        train_loss, f1_train = 0, 0
        all_targets_train, all_predictions_train = [], []

        acc_train, f1_train, train_loss = train_loop(model, optimizer, criterion, train_loader, device, args, epoch)

        if args.use_scheduler: scheduler.step()
        
        if args.skip_validation: continue

        # evaluation loop
        val_loss, val_accuracy, val_f1, _, _, _ = eval_loop(model, criterion, val_loader, device, num_classes)
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

        # Save the best model based on validation F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict()

        # Log epoch metrics to wandb
        if args.log: wandb.log({
            "train_loss": train_loss,
            "train_accuracy": acc_train,
            "train_f1": f1_train,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
        })

# Load the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)

print("Finished Training")

# testing on the test set
test_loss, test_accuracy, test_f1, test_sensitivity, test_ppv, test_specificity = eval_loop(model, criterion, test_loader, device, num_classes)

# print test metrics
print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
print_metrics_table(test_sensitivity, test_ppv, test_specificity, class_names=args.classes)

if args.log:
    # add table data
    table = wandb.Table(
        data=[test_sensitivity, test_ppv, test_specificity], 
        columns=args.classes, 
        rows=["Sensitivity", "PPV", "Specificity"]
    )
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "summary_table": table
    })

    
