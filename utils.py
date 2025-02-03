
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable  # Import PrettyTable for table formatting
import torch
import numpy as np
import random
from dataset import ECGDataset
from collections import Counter
from models.xLSTM import myxLSTM
from models.simple_LSTM import ECG_LSTM, ECG_CONV1D_LSTM
import torch.nn.functional as F

def calculate_metrics(y_true, y_pred, num_classes):
    """Calculates per-class sensitivity, PPV, and specificity."""

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))  #Important to specify labels

    sensitivity = []
    ppv = []
    specificity = []


    for i in range(num_classes):
      TP = cm[i, i]
      FP = sum(cm[:, i]) - TP
      FN = sum(cm[i, :]) - TP
      TN = sum(sum(cm)) - TP - FP - FN


      # Handle potential division by zero
      sensitivity_i = TP / (TP + FN) if (TP+FN) !=0 else 0.0
      ppv_i = TP / (TP + FP) if (TP + FP) != 0 else 0.0
      specificity_i = TN / (TN + FP) if (TN + FP) !=0 else 0.0

      sensitivity.append(sensitivity_i)
      ppv.append(ppv_i)
      specificity.append(specificity_i)


    return sensitivity, ppv, specificity


def print_metrics_table(sensitivity, ppv, specificity, class_names = [ "N", "S", "V", "F", "Q"] ):
  """Prints a formatted table of per-class metrics."""

  table = PrettyTable()
  table.field_names = ["Class", "Sensitivity", "PPV", "Specificity"]

  for i, class_name in enumerate(class_names):
      table.add_row([class_name, f"{sensitivity[i]:.4f}", f"{ppv[i]:.4f}", f"{specificity[i]:.4f}"])

  print(table)

def setup_determinitic_seeds():
    """
    Sets up deterministic seeds for reproducibility.
    """
    # deterministic seeds 
    print("Setting deterministic mode")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    g = torch.Generator()
    g.manual_seed(0)
    return g

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_datasets(args, train_transform):
  """
  Returns the train, val, and test datasets.
  """
  
  # create the datasets
  train_dataset = ECGDataset(f'data/preprocessed_{args.dataset}', subset='train', transform=train_transform, classes=args.classes, num_leads=args.num_leads)
  val_dataset = ECGDataset(f'data/preprocessed_{args.dataset}', subset='val', transform=train_transform, classes=args.classes, num_leads=args.num_leads)
  test_dataset = ECGDataset(f'data/preprocessed_{args.dataset}', subset='test', transform=None, classes=args.classes, num_leads=args.num_leads)

  print(f"Test Dataset len: {len(test_dataset)}")
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

  return train_dataset, val_dataset, test_dataset

def get_training_class_weights(train_dataset, num_classes):
  """
  Returns the class weights for the training dataset.
  """
  # 2. Instantiate Model, Loss, and Optimizer
  labels = [label for _, label, _ in train_dataset]
  # labels = train_dataset.get_labels()
  class_counts = Counter(labels)
  print(f"Class Counts: {class_counts}")
  total_samples = len(labels)
  num_classes = len(class_counts)

  class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
  weights = torch.tensor([class_weights[cls] for cls in range(num_classes)], dtype=torch.float32)
  print(f"Class Weights: {weights}")
  return weights
   

def get_model(args):
  if args.model == 'xLSTM':
      model = myxLSTM(args.num_leads, num_classes=len(args.classes), dropout=args.dropout, xlstm_depth=args.num_layers, activation_fn=args.act_fn, pooling=args.pooling, num_leads=args.num_leads)
  elif args.model == 'LSTM':
      model = ECG_LSTM(args.input_size, args.hidden_size, args.num_layers, len(args.classes), args.dropout)
  elif args.model == 'CONV1D_LSTM':
      model = ECG_CONV1D_LSTM(args.input_size, args.hidden_size, args.num_layers, len(args.classes), args.dropout)
  else:
      raise ValueError("Model not supported")
  return model


def contrastive_coupled_loss(outputs, labels, patient_ids, class_weights, margin=.1):
    outputs = F.normalize(outputs, p=2, dim=1) # Normalize embeddings
    # outer product on the outputs to get cosine similarity
    similarity_matrix = torch.mm(outputs, outputs.t()) - torch.eye(outputs.size(0)).to(outputs.device)
    # create a matrix where the same patient has a 0 and different patients have a 1
    patient_matrix = (patient_ids.unsqueeze(0) != patient_ids.unsqueeze(1)).float().to(outputs.device) 

    # set to zero values between same patients
    similarity_matrix = similarity_matrix * patient_matrix

    # get the matrix for same label (1) and different label (0)
    label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(outputs.device)
    # apply class weights in both dimensions
    weight_matrix = class_weights[labels].unsqueeze(1) * class_weights[labels].unsqueeze(0)

    positive_loss = torch.clamp(margin - similarity_matrix, min=0) * label_matrix * weight_matrix # Positive pairs: maximize similarity
    negative_loss = torch.clamp(similarity_matrix + margin, min=0) * (1 - label_matrix)  # Negative pairs: minimize similarity

    # maximize the similarity between same label and minimize the similarity between different labels
    loss = (positive_loss + negative_loss).sum(dim=1).mean()
    return loss

def contrastive_cluster_loss(outputs, patient_ids):
    # outer product on the outputs to get cosine similarity
    similarity_matrix = torch.mm(outputs, outputs.t()) - torch.eye(outputs.size(0)).to(outputs.device)
    # create a matrix where the same patient has a 0 and different patients have a 1
    patient_matrix = (patient_ids.unsqueeze(0) == patient_ids.unsqueeze(1)).float()
    # set to zero values between same patients
    similarity_matrix = similarity_matrix * patient_matrix

    similarity_matrix = similarity_matrix.abs()
    loss = -similarity_matrix.sum(dim=1).mean()
    return loss

   