import torch
import warnings
warnings.filterwarnings('always')
from dataset import ECGDataset, collate_fn
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
from utils import *
warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils import calculate_metrics, print_metrics_table

# Argument parser for hyperparameters
parser = argparse.ArgumentParser(description='Train ECG model')
parser.add_argument('--input_size', type=int, default=2, help='Input size')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.00001, help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
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
parser.add_argument('--classes', nargs='*', type=str, default=['N', 'S', 'V', 'F', 'Q'], help='Classes')
parser.add_argument('--num_leads', type=int, default=2, help='Number of leads')
parser.add_argument('--include_validation', action='store_true', help='Include validation set', default=False)
parser.add_argument('--label_smoothing', type=float, default=.0, help='Label smoothing')
parser.add_argument('--pretrain', action='store_true', help='Pretrain model')
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
train_dataset, val_dataset, test_dataset = get_datasets(args, transform)

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

if args.pretrain:
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=args.wd)  # Use appropriate optimizer
    # optimizer with all but not fc
    optimizer_encoder = optim.AdamW([
        param for name, param in model.named_parameters() if 'fc' not in name
    ], lr=args.lr, weight_decay=args.wd)  # Use appropriate optimizer
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

for epoch in range(args.epochs):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    train_loss, f1_train = 0, 0
    all_targets_train, all_predictions_train = [], []
    if args.pretrain:
        pretrain_loss = 0

    for batch_idx, (data, target, lengths, patient_ids) in loop:
        data = data.to(device).float()
        target = target.to(device)

        if args.pretrain:
            optimizer_encoder.zero_grad()
            embeddings = model.get_embeddings(data)
            p_loss = contrastive_coupled_loss(embeddings, target, patient_ids, weights)
            p_loss.backward()
            #print(loss.item())
            optimizer_encoder.step()
            pretrain_loss += p_loss.item()

        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data, lengths)
        loss = criterion(outputs, target)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        all_targets_train.extend(target.cpu().numpy())
        all_predictions_train.extend(predicted.cpu().numpy())

        loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
        if args.pretrain:
            loop.set_postfix(loss=loss.item(), pretrain_loss=p_loss.item())
        else:
            loop.set_postfix(loss=loss.item())

    # calculate metrics
    train_loss /= len(train_loader)
    pretrain_loss /= len(train_loader)
    accuracy_train = accuracy_score(all_targets_train, all_predictions_train)
    f1_train = f1_score(all_targets_train, all_predictions_train, average='macro', zero_division=0)
    train_sensitivity, train_ppv, train_specificity = calculate_metrics(all_targets_train, all_predictions_train, num_classes)

    if args.pretrain:
        print(f"Epoch [{epoch+1}/{args.epochs}] Training Loss: {train_loss:.4f}, Pretrain Loss: {pretrain_loss:.4f}, Accuracy: {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")
    else:
        print(f"Epoch [{epoch+1}/{args.epochs}] Training Loss: {train_loss:.4f}, Accuracy: {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")

    if args.use_scheduler: scheduler.step()
    
    if args.skip_validation: continue

    # Validation loop (add validation metrics calculation here)
    model.eval()
    val_loss = 0
    all_targets, all_predictions = [], []
    with torch.no_grad():
        val_loop = tqdm(val_loader, total=len(val_loader), leave=False)
        for data, target, lengths, patient_ids in val_loop:
            data = data.to(device).float()
            target = target.to(device)

            outputs = model(data, lengths)
            loss = criterion(outputs, target)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # calculate metrics
    val_loss /= len(val_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    val_sensitivity, val_ppv, val_specificity = calculate_metrics(all_targets, all_predictions, num_classes)

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # Save the best model based on validation F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model_state = model.state_dict()

    # Log epoch metrics to wandb
    if args.log: 
        log = {
            "train_loss": train_loss,
            "train_accuracy": accuracy_train,
            "train_f1": f1_train,
            "val_loss": val_loss,
            "val_accuracy": accuracy,
            "val_f1": f1,
        }
        if args.pretrain:
            log["pretrain_loss"] = pretrain_loss
        wandb.log(log)

# Load the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)

print("Finished Training")

# Evaluation on the Test Set (after training)
model.eval()
with torch.no_grad():
    test_loss = 0
    all_targets, all_predictions = [], []
    for data, target, lengths, patient_ids in tqdm(test_loader):
        data = data.to(device).float()
        target = target.to(device)

        outputs = model(data, lengths)
        loss = criterion(outputs, target)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# calculate metrics
test_loss /= len(test_loader)
test_accuracy = accuracy_score(all_targets, all_predictions)
test_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
test_sensitivity, test_ppv, test_specificity = calculate_metrics(all_targets, all_predictions, num_classes)

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

    
