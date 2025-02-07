from utils.loss_utils import contrastive_coupled_loss, contrastive_cluster_loss, sparsity_loss
from utils.utils import calculate_metrics
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch

def pretrain_step(model, optimizer, data, target, patient_ids, weights, args, is_eval=False):
    """
    Pretrain step for the model.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        data (torch.Tensor): The input data.
        target (torch.Tensor): The target labels.
        patient_ids (torch.Tensor): The patient IDs.
        weights (torch.Tensor): The class weights.
        args (argparse.Namespace): The command-line arguments.
    
    Returns:
        float: The loss of the model on the training data.
    """
    if not is_eval: optimizer.zero_grad()

    embeddings = model.get_embeddings(data)
    # calculate the contrastive loss
    if args.cluster_loss:
        p_loss = contrastive_cluster_loss(embeddings, patient_ids)
    else:
        p_loss = contrastive_coupled_loss(embeddings, target, patient_ids, weights)

    # if needed, add sparsity loss
    if args.sparse_loss:
        spars_loss = sparsity_loss(embeddings, k=0.2)
        p_loss = p_loss + 0.1 * spars_loss

    if not is_eval:
        # do the backward pass  
        p_loss.backward()
        optimizer.step()

    return p_loss.item()

def train_step(model, optimizer, criterion, data, target):
    """
    Training step for the model.
    
    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (torch.nn.Module): The loss function.
        data (torch.Tensor): The input data.
        target (torch.Tensor): The target labels.
    
    Returns:
        tuple: A tuple containing the following training metrics:
            - loss (float): The loss of the model on the training data.
            - outputs (torch.Tensor): The model outputs.
            - predicted (torch.Tensor): The predicted labels.
    """
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
    _, predicted = torch.max(outputs, 1)
    return loss.item(), predicted

def eval_step(model, criterion, data, target):
    """
    Evaluates the model on the given data.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): The loss function.
        data (torch.Tensor): The input data.
        target (torch.Tensor): The target labels.
    
    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - loss (float): The average loss over the evaluation data.
            - outputs (torch.Tensor): The model outputs.
            - predicted (torch.Tensor): The predicted labels.
    """
    outputs = model(data)
    loss = criterion(outputs, target)
    _, predicted = torch.max(outputs, 1)
    return loss.item(), outputs, predicted

def eval_loop(model, criterion, dataloader, device, num_classes):
    """
    Evaluates the model on the given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): The loss function.
        dataloader (torch.utils.data.DataLoader): The dataloader providing the evaluation data.
        device (torch.device): The device to run the evaluation on (e.g., 'cpu' or 'cuda').
        num_classes (int): The number of classes in the dataset.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - loss (float): The average loss over the evaluation dataset.
            - accuracy (float): The accuracy of the model on the evaluation dataset.
            - f1 (float): The macro-averaged F1 score of the model on the evaluation dataset.
            - sensitivity (float): The sensitivity (recall) of the model on the evaluation dataset.
            - ppv (float): The positive predictive value (precision) of the model on the evaluation dataset.
            - specificity (float): The specificity of the model on the evaluation dataset.
    """
    model.eval()
    with torch.no_grad():
        loss = 0
        all_targets, all_predictions = [], []
        eval_loop = tqdm(dataloader, total=len(dataloader), leave=False)
        for data, target, _, _ in eval_loop:
            data = data.to(device).float()
            target = target.to(device)

            ev_loss, _, predicted = eval_step(model, criterion, data, target)
            loss += ev_loss

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # calculate metrics
        loss /= len(dataloader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        acc_avg, acc, sensitivity, specificity, ppv = calculate_metrics(all_targets, all_predictions, num_classes)

        return loss, accuracy, f1, sensitivity, ppv, specificity

def train_loop(model, optimizer, criterion, train_loader, device, args, epoch):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    train_loss = 0
    all_targets_train, all_predictions_train = [], []
    
    for _, (data, target, _, _) in loop:
        data = data.to(device).float()
        target = target.to(device)


        t_loss, predicted = train_step(model, optimizer, criterion, data, target)
        train_loss += t_loss

        all_targets_train.extend(target.cpu().numpy())
        all_predictions_train.extend(predicted.cpu().numpy())

        loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
        loop.set_postfix(loss=t_loss)

    # calculate metrics
    train_loss /= len(train_loader)
    accuracy_train = accuracy_score(all_targets_train, all_predictions_train)
    f1_train = f1_score(all_targets_train, all_predictions_train, average='macro', zero_division=0)
    # train_sensitivity, train_ppv, train_specificity = calculate_metrics(all_targets_train, all_predictions_train, num_classes)

    print(f"Epoch [{epoch+1}/{args.epochs}] Training Loss: {train_loss:.4f}, Accuracy: {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")
    return accuracy_train, f1_train, train_loss

def pretrain_loop(model, optimizer, criterion, train_loader, device, args, epoch, weights):
    """
    Training loop for the pretraining phase. If separate_pretrain is set to True, only the pretraining loss is calculated.
    Otherwise, the training step is done here where one step is made with the contrastive loss and one step with the classification loss.
    """
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    pretrain_loss = 0
    if not args.separate_pretrain:
        train_loss, f1_train = 0, 0
        all_targets_train, all_predictions_train = [], []

    for _, (data, target, _, patient_ids) in loop:
        data = data.to(device).float()
        target = target.to(device)

        p_loss = pretrain_step(model, optimizer, data, target, patient_ids, weights, args)
        pretrain_loss += p_loss

        if not args.separate_pretrain:
            t_loss, predicted = train_step(model, optimizer, criterion, data, target)
            train_loss += t_loss

            all_targets_train.extend(target.cpu().numpy())
            all_predictions_train.extend(predicted.cpu().numpy())

            loop.set_description(f"Epoch [{epoch+1}/{args.epochs_pretrain}]")
            loop.set_postfix(loss=t_loss, pretrain_loss=p_loss)
        else:
            loop.set_description(f"Epoch [{epoch+1}/{args.epochs_pretrain}]")
            loop.set_postfix(pretrain_loss=p_loss)

    pretrain_loss /= len(train_loader)

    if not args.separate_pretrain:
        train_loss /= len(train_loader)
        accuracy_train = accuracy_score(all_targets_train, all_predictions_train)
        f1_train = f1_score(all_targets_train, all_predictions_train, average='macro', zero_division=0)

        print(f"Epoch [{epoch+1}/{args.epochs}] Training Loss: {train_loss:.4f}, Pretrain Loss: {pretrain_loss:.4f} Accuracy: {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")
        return {
            "accuracy_train": accuracy_train,
            "f1_train": f1_train,
            "train_loss": train_loss,
            "pretrain_loss": pretrain_loss
        }
    else:
        print(f"Epoch [{epoch+1}/{args.epochs_pretrain}] Pretrain Loss: {pretrain_loss:.4f}")
        return {
            "pretrain_loss": pretrain_loss
        }

def eval_pretrain_loop(model, dataloader, device, weights, args):
    model.eval()
    loop = tqdm(dataloader, total=len(dataloader), leave=False)
    with torch.no_grad():
        loss = 0
        for data, target, _, patient_ids in loop:
            data = data.to(device).float()
            target = target.to(device)

            ev_loss = pretrain_step(model, None, data, target, patient_ids, weights, args, is_eval=True)
            loss += ev_loss

    loss /= len(dataloader)
    print(f"Validation pretrain Loss: {loss:.4f}")
    return loss

