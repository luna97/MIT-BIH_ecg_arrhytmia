
import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_coupled_loss(outputs, labels, patient_ids, class_weights, margin=.1):
    """
    Computes the contrastive coupled loss for a batch of embeddings.

    The loss is calculated by first normalizing the output embeddings and then computing a similarity matrix using the cosine similarity. 
    A patient matrix is created to differentiate between samples from the same and different patients. 
    The similarity matrix is adjusted to zero out values between the same patients. 
    A label matrix is created to differentiate between samples with the same and different labels. 
    Class weights are applied to the label matrix to account for class imbalance.

    The positive loss is calculated by clamping the margin minus the similarity matrix, multiplied by the label matrix and the weight matrix. 
    This encourages the model to maximize the similarity between samples with the same label. 
    The negative loss is calculated by clamping the similarity matrix plus the margin, multiplied by the inverse of the label matrix. 
    This encourages the model to minimize the similarity between samples with different labels.

    The final loss is the sum of the positive and negative losses, averaged over the batch.

    Args:
        outputs (torch.Tensor): The output embeddings from the model, shape (batch_size, embedding_dim).
        labels (torch.Tensor): The ground truth labels for the batch, shape (batch_size,).
        patient_ids (torch.Tensor): The patient IDs corresponding to each sample in the batch, shape (batch_size,).
        class_weights (torch.Tensor): The weights for each class, shape (num_classes,).
        margin (float, optional): The margin value for the contrastive loss. Default is 0.1.

    Returns:
        torch.Tensor: The computed contrastive coupled loss.
    """
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

def sparsity_loss(outputs, k=0.1, p=2):
    """
    Computes the sparsity loss for the given outputs by keeping the top k% of the embeddings
    and setting the rest to zero. The loss is calculated as the mean squared value of the masked outputs.

    Args:
        outputs (torch.Tensor): The input tensor containing the embeddings.
        k (float, optional): The percentage of top embeddings to keep. Default is 0.1 (10%).

    Returns:
        torch.Tensor: The computed sparsity loss.
    """
    # keep the top k% of the embeddings and set the rest to zero
    idxs = torch.topk(outputs, int(k * outputs.size(1)), dim=1, largest=True)[1]
    mask = torch.ones_like(outputs)
    mask.scatter_(1, idxs, value=0)
    loss = (outputs * mask).pow(p).mean()
    return loss

def contrastive_cluster_loss(outputs, patient_ids, margin=0.5):
    outs = F.normalize(outputs, p=2, dim=1) # Normalize embeddings
    similarity_matrix = torch.mm(outs, outs.t()) - torch.eye(outputs.size(0)).to(outputs.device)

    # create a matrix where the same patient has a 0 and different patients have a 1
    patient_matrix = (patient_ids.unsqueeze(0) == patient_ids.unsqueeze(1)).float().to(outputs.device)
    # set to zero values between same patients
    similarity_matrix = similarity_matrix * patient_matrix

    losses = torch.clamp(similarity_matrix + margin, min=0)
    loss = (losses).sum(dim=1).mean()

    return loss

