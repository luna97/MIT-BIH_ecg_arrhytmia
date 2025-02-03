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
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils import calculate_metrics, print_metrics_table

# Argument parser for hyperparameters
parser = argparse.ArgumentParser(description='Train ECG model')
parser.add_argument('--input_size', type=int, default=2, help='Input size')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.00001, help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--augment_data', action='store_true', help='Augment data')
parser.add_argument('--model', type=str, default='xLSTM', help='Model to use')
parser.add_argument('--skip_validation', action='store_true', help='Skip validation')
parser.add_argument('--dataset', type=str, default='nk_clean', help='Dataset to use')
parser.add_argument('--deterministic', action='store_true', help='Deterministic', default=False)
parser.add_argument('--log', action='store_true', help='Log to wandb')
parser.add_argument('--run_name', type=str, default=None, help='Run name')
parser.add_argument('--use_scheduler', action='store_true', help='Use scheduler')
parser.add_argument('--pooling', type=str, default='max', help='Pooling method')
parser.add_argument('--act_fn', type=str, default='leakyrelu', help='Activation function')
parser.add_argument('--classes', nargs='*', type=str, default=['N', 'S', 'V', 'F', 'Q'], help='Classes')
parser.add_argument('--num_leads', type=int, default=2, help='Number of leads')
parser.add_argument('--include_validation', action='store_true', help='Include validation set', default=False)
parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')

args = parser.parse_args()

if args.deterministic:
    # deterministic seeds 
    print("Setting deterministic mode")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)


if not args.augment_data:
    transform = None
else: 
    transform = transforms.Compose([
        CropResizing(start_idx=0, resize=False),
        FTSurrogate(phase_noise_magnitude=0.075, prob=0.5),
        Jitter(sigma=0.2),
        Rescaling(sigma=0.5)
    ])

# create the dataloaders
train_dataset = ECGDataset(f'data/preprocessed_{args.dataset}', subset='train', transform=transform, classes=args.classes, num_leads=args.num_leads)
val_dataset = ECGDataset(f'data/preprocessed_{args.dataset}', subset='val', transform=transform, classes=args.classes, num_leads=args.num_leads)
test_dataset = ECGDataset(f'data/preprocessed_{args.dataset}', subset='test', transform=None, classes=args.classes, num_leads=args.num_leads)

print(f"Test Dataset: {len(test_dataset)}")
if args.include_validation:
    print('Mixing val and train DS1 datasets')
    # get one dataset from train and val
    train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    train_len = int(0.8 * len(train_val_dataset))
    val_len = len(train_val_dataset) - train_len
    # split the dataset into train and val
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_len, val_len])
else:
    print('Using only training DS1 dataset')
    train_len = int(0.8 * len(train_dataset))
    val_len = len(train_dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])
    print(f"Train Dataset: {len(train_dataset)}")
    print(f"Val Dataset: {len(val_dataset)}")


if args.deterministic:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)
else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Instantiate Model, Loss, and Optimizer
labels = [label for _, label in train_dataset]
#labels = train_dataset.get_labels()
class_counts = Counter(labels)
print(f"Class Counts: {class_counts}")
total_samples = len(labels)
num_classes = len(class_counts)

class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
weights = torch.tensor([class_weights[cls] for cls in range(num_classes)], dtype=torch.float32)

if args.model == 'xLSTM':
    model = myxLSTM(args.num_leads, num_classes=num_classes, dropout=args.dropout, xlstm_depth=args.num_layers, activation_fn=args.act_fn, pooling=args.pooling, num_leads=args.num_leads)
elif args.model == 'LSTM':
    model = ECG_LSTM(args.input_size, args.hidden_size, args.num_layers, num_classes, args.dropout)
elif args.model == 'CONV1D_LSTM':
    model = ECG_CONV1D_LSTM(args.input_size, args.hidden_size, args.num_layers, num_classes, args.dropout)
else:
    raise ValueError("Model not supported")

criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)  # Use appropriate loss for multi-class classification
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)  # Use appropriate optimizer

# Add Cosine Annealing LR Scheduler
if args.use_scheduler:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # T_max is the number of iterations for the first half of the cycle


if args.log:
    wandb.init(project="ecg-classification", config=args, name=args.run_name)
    wandb.watch(model)
    # log model architecture to wandb
    wandb.config.update({"model_architecture": str(model)})


# 3. Training Loop
model.to(device)

best_f1 = 0.0
best_model_state = None

for epoch in range(args.epochs):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    train_loss = 0
    f1_train = 0
    all_targets_train = []
    all_predictions_train = []

    for batch_idx, (data, target, lengths) in loop:
        data = data.to(device).float()
        target = target.to(device)

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
        loop.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)
    accuracy_train = accuracy_score(all_targets_train, all_predictions_train)
    f1_train = f1_score(all_targets_train, all_predictions_train, average='macro', zero_division=0)
    train_sensitivity, train_ppv, train_specificity = calculate_metrics(all_targets_train, all_predictions_train, num_classes)

    print(f"Epoch [{epoch+1}/{args.epochs}] Training Loss: {train_loss:.4f}, Accuracy: {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")

    if args.use_scheduler:
        # Step the scheduler
        scheduler.step()
    
    if args.skip_validation:
        continue

    # Validation loop (add validation metrics calculation here)
    model.eval()
    val_loss = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        val_loop = tqdm(val_loader, total=len(val_loader), leave=False)
        for data, target, lengths in val_loop:
            data = data.to(device).float()
            target = target.to(device)

            outputs = model(data, lengths)
            loss = criterion(outputs, target)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)

    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    val_sensitivity, val_ppv, val_specificity = calculate_metrics(all_targets, all_predictions, num_classes)

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # Save the best model based on validation F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model_state = model.state_dict()

    if args.log:
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": accuracy_train,
            "train_f1": f1_train,
            "val_loss": val_loss,
            "val_accuracy": accuracy,
            "val_f1": f1,
        })

# Load the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)

print("Finished Training")


#  Evaluation on the Test Set (after training)
model.eval()
with torch.no_grad():
    test_loss = 0
    all_targets = []
    all_predictions = []
    for data, target, lengths in tqdm(test_loader):
        data = data.to(device).float()
        target = target.to(device)

        outputs = model(data, lengths)
        loss = criterion(outputs, target)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

test_loss /= len(test_loader)

test_accuracy = accuracy_score(all_targets, all_predictions)
test_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
test_sensitivity, test_ppv, test_specificity = calculate_metrics(all_targets, all_predictions, num_classes)

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

    
