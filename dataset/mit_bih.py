import torch
import os
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np

class ECGMITBIHDataset(torch.utils.data.Dataset):

    def __init__(self, data_folder, subset='train', num_leads=1, name='t_wave_split', oversample=False, random_shift=False, patch_size=64, normalize=True, nkclean=False):
        self.data_folder = data_folder
        self.subset = subset
        self.samples = []
        self.random_shift = random_shift
        self.nkclean = nkclean
        self.num_leads = num_leads
        self.samples = pd.read_csv(os.path.join(data_folder, name, f'labels_{subset}.csv'))
        if not oversample:
            self.samples = self.samples[self.samples['is_oversampled'] == False]

        self.patch_size = patch_size
        self.normalize = normalize
        
        # get all the different values for column patient
        self.patients = self.samples['patient'].unique()
        self.comments = {}

        # load on memory all the signals
        self.signals = {}
        for patient in self.patients:
            signal, _ = wfdb.rdsamp(os.path.join(data_folder, 'raw', f'{patient}'))
            header = wfdb.rdheader(os.path.join(data_folder, 'raw', f'{patient}'))
            # normalize the signal
            self.signals[patient] = signal
            self.comments[patient] = header.comments


    def __len__(self):
        return len(self.samples)
    
    def get_label_int(self, label):
        if label == 'N': return 0
        if label == 'S': return 1
        if label == 'V': return 2
        if label == 'F': return 3
        if label == 'Q': return 4

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        patient = sample['patient']
        signal = self.signals[patient]
        heartbeat_signal = signal[sample['hb_start']:sample['hb_end'], :self.num_leads]

        if self.random_shift:
            window_start = sample['win_start']
            window_end = sample['win_end']
            shift = torch.randint(- self.patch_size // 3, self.patch_size // 3, (1,)).item() # shift between 0 and patch_size // 3
            window_start += shift
            window_end += shift
            # ensure that the window is inside the signal
            window_start = max(0, window_start)
            window_end = min(window_end, len(signal))
            window_signal = signal[window_start:window_end, :self.num_leads]
        else:
            window_signal = signal[sample['win_start']:sample['win_end'], :self.num_leads]

        if window_signal.shape[0] == 0:
            print(f'index {idx}, patient {patient}, window_signal shape {window_signal.shape}, heartbeat_signal shape {heartbeat_signal.shape}')

        if self.normalize:
            if window_signal.std(axis=0) != 0:
                window_signal = (window_signal - window_signal.mean(axis=0)) / window_signal.std(axis=0)
            if heartbeat_signal.std(axis=0) != 0:
                heartbeat_signal = (heartbeat_signal - heartbeat_signal.mean(axis=0)) / heartbeat_signal.std(axis=0)

        if self.nkclean:
            window_signal[:, 0] = nk.ecg_clean(window_signal[:, 0], sampling_rate=360)
            heartbeat_signal[:, 0] = nk.ecg_clean(heartbeat_signal[:, 0], sampling_rate=360)

        comment = self.comments[int(patient)][0].split(' ')
        age = min(0, int(comment[0]))
        is_male = comment[1] == 'M'

        # create pandas row with the tabular data
        tab_data = pd.DataFrame({
            'age': [age],
            'is_male': [is_male],
            'RBBB': sample['orig_label'] == 'R',
            'LBBB': sample['orig_label'] == 'L',
        })

        return {
            'heartbeat': torch.tensor(heartbeat_signal, dtype=torch.float32),
            'signal': torch.tensor(window_signal, dtype=torch.float32),
            'label': self.get_label_int(sample['label']),
            'tab_data': tab_data
        }
    
    def split_validation_training(self, val_size=0.2):
        if self.subset != 'train':
            raise ValueError('Can only split the training dataset')

        count = 0
        ids_val, ids_train = [], []
        for patient in self.patients:
            df_patient = self.samples[self.samples['patient'] == patient]
            num_samples = len(df_patient)
            val_len = int(val_size * num_samples)

            # the first 0.8 of the patients go to the training set
            ids_train.extend(range(count, count + num_samples - val_len))
            ids_val.extend(range(count + num_samples - val_len, count + num_samples))
            count += num_samples
        return torch.utils.data.Subset(self, ids_train), torch.utils.data.Subset(self, ids_val)

    
def collate_fn(batch):
    signals = [item['signal'] for item in batch]
    hb = [item['heartbeat'] for item in batch]
    mask_signals = [torch.ones_like(item['signal'], dtype=torch.bool) for item in batch]
    mask_hb = [torch.ones_like(item['heartbeat'], dtype=torch.bool) for item in batch]

    # pad to same length and pad to match the patch size module
    heartbeat_signals = torch.nn.utils.rnn.pad_sequence(hb, batch_first=True)

    window_signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
    labels = torch.tensor([item['label'] for item in batch])
    mask_signals = torch.nn.utils.rnn.pad_sequence(mask_signals, batch_first=True)
    mask_hb = torch.nn.utils.rnn.pad_sequence(mask_hb, batch_first=True)
    # print('signals shape', window_signals.shape)
    tab_data = pd.concat([item['tab_data'] for item in batch], axis=0)
    # print('tab_data type mitbih', type(tab_data))

    return {
        'heartbeat': heartbeat_signals,
        'mask_hb': mask_hb,
        'signal': window_signals,
        'mask': mask_signals,
        'label': labels,
        'tab_data': tab_data
    }
