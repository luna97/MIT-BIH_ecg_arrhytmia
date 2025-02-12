
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable  # Import PrettyTable for table formatting
import torch
import numpy as np
import random
from old.dataset_old import ECGDataset
from collections import Counter
from models.xLSTM import myxLSTM
from models.simple_LSTM import ECG_LSTM, ECG_CONV1D_LSTM
from models.seq2seq import Seq2SeqModel
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def calculate_metrics(y_true, y_pred, num_classes):
    """Calculates per-class sensitivity, PPV, and specificity."""

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))  #Important to specify labels

    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    # Sensitivity, hit rate, recall, or true positive rate
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    ACC_macro = np.mean(
        ACC)  # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)

    return ACC_macro, ACC, TPR, TNR, PPV

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

def get_datasets(args, train_transform, data_dir):
  """
  Returns the train, val, and test datasets.
  """
  
  # create the datasets
  train_dataset = ECGDataset(f'{data_dir}/preprocessed_{args.dataset}', subset='train', transform=train_transform, classes=args.classes, num_leads=args.num_leads)
  val_dataset = ECGDataset(f'{data_dir}/preprocessed_{args.dataset}', subset='val', transform=train_transform, classes=args.classes, num_leads=args.num_leads)
  test_dataset = ECGDataset(f'{data_dir}/preprocessed_{args.dataset}', subset='test', transform=None, classes=args.classes, num_leads=args.num_leads)

  print(f"Test Dataset len: {len(test_dataset)}")
  if args.include_val:
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
        model = myxLSTM(args.num_leads, num_classes=len(args.classes), dropout=args.dropout, xlstm_depth=args.num_layers, activation_fn=args.act_fn, pooling=args.pooling, num_leads=args.num_leads, channels=[128])
    elif args.model == 'LSTM':
        model = ECG_LSTM(args.input_size, args.hidden_size, args.num_layers, len(args.classes), args.dropout)
    elif args.model == 'CONV1D_LSTM':
        model = ECG_CONV1D_LSTM(args.input_size, args.hidden_size, args.num_layers, len(args.classes), args.dropout)
    elif args.model == 'seq2seq':
        model = Seq2SeqModel(args.input_size, args.hidden_size, args.num_layers, len(args.classes), args.dropout)
    else:
        raise ValueError("Model not supported")
    return model
