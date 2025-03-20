
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable  # Import PrettyTable for table formatting
import torch
import numpy as np
import random
from collections import Counter
from models.xLSTM import myxLSTM
from models.simple_LSTM import ECG_LSTM, ECG_CONV1D_LSTM
from models.seq2seq import Seq2SeqModel
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

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


def get_training_class_weights(train_dataset, do_not_consider_classes = []):
  """
  Returns the class weights for the training dataset.
  """
  # 2. Instantiate Model, Loss, and Optimizer
  labels = [sample['label'] for sample in train_dataset]

  # remove classes that should not be considered
  labels = [label for label in labels if label not in do_not_consider_classes]
  
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
