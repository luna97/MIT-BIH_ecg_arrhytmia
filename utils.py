
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable  # Import PrettyTable for table formatting

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


