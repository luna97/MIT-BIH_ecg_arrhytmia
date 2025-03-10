import torch

def masked_mse_loss(input, target, reduction='mean'):
    out = (input - target)**2
    # do not consider elements to 0 from the input
    out = out[target != 0]
    if reduction == "mean":
        return out.mean()
    elif reduction == "none":
        return out
    
def masked_min_max_loss(input, target, reduction='mean', patch_size=100):
    # input should be tokenized
    batch_size, sig_len, num_channels = input.shape
    #print('batch_size', batch_size)
    #print('sig_len', sig_len)

    tokenized_inp = input.view(batch_size, sig_len // patch_size, patch_size, num_channels)
    tokenized_target = target.view(batch_size, sig_len // patch_size, patch_size, num_channels)   
    # find the min and max value of each patch
    min_inp, _ = tokenized_inp.min(dim=-1)
    # print('min_inp', min_inp.shape)
    max_inp, _ = tokenized_inp.max(dim=-1)
    # print('max_inp', max_inp)
    min_target, _ = tokenized_target.min(dim=-1)
    # print('min_target', min_target)
    max_target, _ = tokenized_target.max(dim=-1)
    # print('max_target', max_target)
    # calculate the loss
    out = (min_inp - min_target)**2 + (max_inp - max_target)**2

    # do not consider elements set to 0
    # out = out[target != 0]

    if reduction == "mean":
        return out.mean() / patch_size
    elif reduction == "none":
        return out / patch_size

def masked_mae_loss(input, target, reduction='mean'):
    out = torch.abs(input-target)
    # do not consider elements set to 0
    out = out[target != 0]
    if reduction == "mean":
        return out.mean()
    elif reduction == "none":
        return out
    
def gradient_loss(input, target, reduction='mean', p=2):
    input_grad = input[:, 1:] - input[:, :-1]
    target_grad = target[:, 1:] - target[:, :-1]
    out = torch.pow(input_grad - target_grad, p)
    # do not consider elements that was at 0 in the input
    # using [:, :-1] because preserve the order of the elements
    out = out[target[:, :-1] != 0]
    if reduction == "mean":
        return out.mean()
    elif reduction == "none":
        return out