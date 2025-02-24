from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
import os

def plot_reconstruction(sample, model, patch_size, device, logdir, epoch, name):
    tab_data = sample['tab_data']
    signal = sample['signal'].to(device).unsqueeze(0)
    signal = F.pad(signal, (0, 0, 0, patch_size - signal.shape[1] % patch_size))

    reconstruct, _, _, _ = model.reconstruct(signal, tab_data) # [batch_size, seq_len // 64, patch_size]
    shift_x = signal[:, :reconstruct.shape[1]]
    shift_x = shift_x[:, patch_size:].squeeze()


    shift_reconstruct = reconstruct[:, :-patch_size]

    # try to reconstruct one element at a time
    reconstruct = reconstruct.view(1, -1)
    shift_x = shift_x.view(1, -1)
    #print('rec shape', reconstruct.shape)
    #print('shift_x shape', shift_x.shape)


    # plot the original and reconstructed signal
    plt.figure(figsize=(10, 4))
    plt.plot(shift_x.squeeze().cpu().numpy(), label='original')
    plt.plot(shift_reconstruct.squeeze().detach().cpu().numpy(), label='reconstructed')
    # add a pipe avery patch size samples
    for i in range(1, signal.shape[1] // patch_size - 1):
        plt.axvline(x=patch_size*i, color='gray', linestyle='--')

    plt.legend()

    # mkdir if it does not exist
    os.makedirs(f'{logdir}/epoch_{epoch}', exist_ok=True)

    path = f'{logdir}/epoch_{epoch}/reconstruction_{name}.png'
    plt.savefig(path)
    plt.close()
    return path
    
def plot_generation(sample, model, patch_size, device, logdir, epoch, name):
    tab_data = sample['tab_data']
    signal = sample['signal'].to(device).unsqueeze(0)
    prediction_tokens = torch.zeros([1, 0]).to(device)
    for i in range(0, 20):
        if i == 0:
            new_signal = signal
        else:
            flatten_preds = prediction_tokens.view(1, -1)
            new_signal = torch.cat([signal, flatten_preds.unsqueeze(-1)], dim=1)

        r1, _, _, _ = model.reconstruct(new_signal, tab_data=tab_data)
        # print('reconstruct shape', reconstruct.shape)
        # the last token is the prediction
        p1 = r1[:, -patch_size:]

        prediction_tokens = torch.cat([prediction_tokens, p1], dim=1)

    predictions = prediction_tokens.squeeze().view(-1)

    plt.figure(figsize=(10, 4))
    plt.plot(signal[0, :-patch_size].squeeze().cpu().numpy(), label='original')
    plt.plot(range(len(signal[0, :-patch_size]), len(signal[0, :-patch_size]) + len(predictions)), predictions.detach().cpu().numpy(), label='prediction')
    # add a pipe avery 64 samples
    for i in range(0, (len(signal[0]) + len(predictions)) // patch_size -1):
        plt.axvline(x=patch_size*i, color='gray', linestyle='--')
    plt.legend()

    # mkdir if it does not exist
    os.makedirs(f'{logdir}/epoch_{epoch}', exist_ok=True)

    path = f'{logdir}/epoch_{epoch}/generation_{name}.png'
    plt.savefig(path)
    plt.close()
    return path