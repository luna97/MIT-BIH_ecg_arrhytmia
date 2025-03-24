from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os
import numpy as np

color_1 = (50 / 255, 134 / 255, 143 / 255)
color_2 = (207/ 255, 86/ 255, 86/ 255)

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot_reconstruction(sample, model, patch_size, device, logdir, epoch, name):
    with torch.no_grad():
        tab_data = sample['tab_data'].reset_index(drop=True) if 'tab_data' in sample.keys() else None

        signal = sample['signal'].to(device).unsqueeze(0)

        # print('signal shape', signal.shape)
        # print(tab_data)

        orig_signal = signal.clone()
        if len(signal.shape) == 2:
            signal = signal.unsqueeze(-1)

        reconstruct = model.reconstruct(signal, tab_data)
            
        # if reconstruct is a tuple, get the first element
        if isinstance(reconstruct, tuple):
            reconstruct = reconstruct[0]

        shift_x = signal[:, :reconstruct.shape[1]]
        orig_signal = orig_signal[:, :reconstruct.shape[1]]
        shift_x = shift_x[:, patch_size:].squeeze()
        orig_signal = orig_signal[:, patch_size:].squeeze()

        shift_reconstruct = reconstruct[:, :-patch_size]

        # try to reconstruct one element at a time
        reconstruct = reconstruct.view(1, -1, signal.shape[-1])
        shift_x = shift_x.view(1, -1, signal.shape[-1])

        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(signal.shape[-1] // 2, 2)
        gs.update(wspace=0.08, hspace=0.16)

        for i in range(signal.shape[-1]):
            ax = plt.subplot(gs[i % 6, i // 6])
            ax.plot(orig_signal[..., i].cpu().squeeze().numpy(), color=color_1)
            ax.plot(shift_reconstruct[..., i].cpu().squeeze().numpy(), color=color_2)

            # sometimes the signal is zeroed out by the random drop leads
            # so here i just plot the zeroed out signal with a dashed line to know that channel was zeroed out
            has_augmentation = False
            if (shift_x[..., i] != orig_signal[..., i]).any():
                ax.plot(signal[..., patch_size:reconstruct.shape[1], i].cpu().squeeze().numpy(), color='grey', linestyle='--')
                has_augmentation = True
                
            ax.set_title(leads[i])

            # add vertical lines avery patch size
            for j in range(0, shift_x.shape[1], patch_size):
                ax.axvline(j, color='gray', linestyle='--', linewidth=0.5)

            # ax.set_yticks([])
            if i == 0:
                if has_augmentation:
                    ax.legend(['Original', 'Reconstructed', 'Augmented'], loc='upper left')
                else:
                    ax.legend(['Original', 'Reconstructed'], loc='upper left')

            if i == 5 or i == 11:
                ax.set_xticks(np.arange(0, len(shift_x[0]), 360))
                ax.set_xticklabels(np.arange(0, len(shift_x[0]), 360) // 360)
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticks([])

        # mkdir if it does not exist
        os.makedirs(f'{logdir}/epoch_{epoch}', exist_ok=True)

        path = f'{logdir}/epoch_{epoch}/reconstruction_{name}.png'
        plt.savefig(path)
        plt.close()
        return path
    
def plot_generation(sample, model, patch_size, device, logdir, epoch, name):
    with torch.no_grad():
        tab_data = sample['tab_data'].reset_index(drop=True) if 'tab_data' in sample.keys() else None
        signal = sample['signal'].to(device).unsqueeze(0)

        signal = signal[:, :signal.shape[1] - signal.shape[1] % patch_size]
        if len(signal.shape) == 2:
            signal = signal.unsqueeze(-1)

        generated = model.generate(signal, tab_data=tab_data, length=(2048 // patch_size))

        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(signal.shape[-1] // 2, 2)
        gs.update(wspace=0.08, hspace=0.16)

        for i in range(signal.shape[-1]):
            ax = plt.subplot(gs[i % 6, i // 6])
            ax.plot(signal[..., i].cpu().squeeze().numpy(), color=color_1)
            ax.plot(
                range(signal.shape[1], signal.shape[1] + generated.shape[1]),
                generated[..., i].cpu().squeeze().numpy(), color=color_2)
            ax.set_title(leads[i],)

            # add vertical lines avery patch size
            for j in range(0, signal.shape[1] + generated.shape[1], patch_size):
                ax.axvline(j, color='gray', linestyle='--', linewidth=0.5)

            # ax.set_yticks([])
            if i == 0:
                ax.legend(['Original', 'Generated'], loc='upper left')

            if i == 5 or i == 11:
                ax.set_xticks(np.arange(0, len(signal[0]), 360))
                ax.set_xticklabels(np.arange(0, len(signal[0]), 360) // 360)
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticks([])

        # mkdir if it does not exist
        os.makedirs(f'{logdir}/epoch_{epoch}', exist_ok=True)

        path = f'{logdir}/epoch_{epoch}/generation_{name}.png'
        plt.savefig(path)
        plt.close()
        return path


