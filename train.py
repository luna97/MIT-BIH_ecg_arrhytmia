import torch
from dataset import ECGDataset, collate_fn
from models.simple_LSTM import ECG_LSTM
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score


# create the dataloaders
train_dataset = ECGDataset('data/preprocessed_base', subset='train')
test_dataset = ECGDataset('data/preprocessed_base', subset='test')

# divide training in validation and training
val_len = int(0.2 * len(train_dataset))
train_len = len(train_dataset) - val_len
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])
                                                           
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# 2. Instantiate Model, Loss, and Optimizer
input_size = 2  # Assuming single-lead ECG data
hidden_size = 128  # Adjust as needed
num_layers = 2
num_classes = 5 # Replace with the actual number of arrhythmia classes in your dataset
learning_rate = 0.0003
dropout = 0.3

model = ECG_LSTM(input_size, hidden_size, num_layers, num_classes, dropout)
criterion = nn.CrossEntropyLoss()  # Use appropriate loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 3. Training Loop
num_epochs = 50  # Adjust as needed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    train_loss = 0
    f1_train = 0
    total_train = 0
    correct_train = 0
    all_targets_train = []
    all_predictions_train = []

    for batch_idx, (data, target, lengths) in loop:
        data = data.to(device).float()
        target = target.to(device)
        lengths = lengths.to(device)

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
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()

        all_targets_train.extend(target.cpu().numpy())
        all_predictions_train.extend(predicted.cpu().numpy())

        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)
    accuracy_train = correct_train / total_train
    f1_train = f1_score(all_targets_train, all_predictions_train, average='micro')

    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {train_loss:.4f}, Accuracy: {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")

    

    # Validation loop (add validation metrics calculation here)
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        val_loop = tqdm(val_loader, total=len(val_loader), leave=False)
        for data, target, lengths in val_loop:
            data = data.to(device).float()
            target = target.to(device)
            lengths = lengths.to(device)

            outputs = model(data, lengths)
            loss = criterion(outputs, target)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += target.size(0)
            correct_val += (predicted == target).sum().item()

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())


    val_loss /= len(val_loader)
    accuracy = correct_val / total_val

    f1 = f1_score(all_targets, all_predictions, average='micro')

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            


print("Finished Training")


#  Evaluation on the Test Set (after training)
model.eval()
with torch.no_grad():
    # ... calculate test set metrics ...
    pass