from utils.loss_utils import contrastive_coupled_loss, contrastive_cluster_loss, sparsity_loss
from utils.utils import calculate_metrics
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch

def masked_mse_loss(input, target, mask, reduction='mean'):
    out = (input - target)**2
    out = out * mask
    # do not consider elements set to 0
    out = out[out != 0]
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out
    

def masked_mae_loss(input, target, mask, reduction='mean'):
    out = torch.abs(input-target)
    out = out * mask
    # do not consider elements set to 0
    out = out[out != 0]
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out
    

def gradient_loss(input, target, mask, reduction='mean', p=2):
    mask = mask[:, 1:]
    input_grad = input[:, 1:] - input[:, :-1]
    target_grad = target[:, 1:] - target[:, :-1]
    out = torch.pow(input_grad - target_grad, p)
    out = out * mask
    out = out[out != 0]
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out