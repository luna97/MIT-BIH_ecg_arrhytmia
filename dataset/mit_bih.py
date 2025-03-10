import torch
import os
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
conversion = {
    'MLII' : 'II',
}

class ECGMITBIHDataset(torch.utils.data.Dataset):

    def __init__(self, config, subset='train', name='t_wave_split'):
        self.data_folder = config.data_folder_mit
        self.subset = subset
        self.samples = []
        self.random_shift = config.random_shift
        self.nkclean = config.nk_clean
        self.use_tab_data = config.use_tab_data
        self.patch_size = config.patch_size
        self.normalize = config.normalize

        self.leads_to_use = leads if config.leads == ['*'] else config.leads

        self.samples = pd.read_csv(os.path.join(self.data_folder, name, f'labels_{subset}.csv'))
        # ensure no Nan values
        self.samples['extra_annotations'] = self.samples['extra_annotations'].fillna('')

        if not config.oversample:
            self.samples = self.samples[self.samples['is_oversampled'] == False]

        # get all the different values for column patient
        self.patients = self.samples['patient'].unique()
        self.headers = {}

        # load on memory all the signals
        self.signals = {}
        for patient in self.patients:
            signal, _ = wfdb.rdsamp(os.path.join(self.data_folder, 'raw', f'{patient}'))
            header = wfdb.rdheader(os.path.join(self.data_folder, 'raw', f'{patient}'))
            # normalize the signal
            self.signals[patient] = signal
            self.headers[patient] = header

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
        signal = torch.tensor(signal, dtype=torch.float32)
        header = self.headers[patient]
        heartbeat_signal = signal[sample['hb_start']:sample['hb_end']]

        if self.random_shift:
            window_start = sample['win_start']
            window_end = sample['win_end']
            shift = torch.randint(- self.patch_size // 3, self.patch_size // 3, (1,)).item() # shift between 0 and patch_size // 3
            window_start += shift
            window_end += shift
            # ensure that the window is inside the signal
            window_start = max(0, window_start)
            window_end = min(window_end, len(signal))
            window_signal = signal[window_start:window_end]
        else:
            window_signal = signal[sample['win_start']:sample['win_end']]

        if self.normalize:
            std = window_signal.std(axis=(0, -1))
            std[std == 0] = 1 # avoid division by zero, samples with std = 0 are all zero
            window_signal = (window_signal - window_signal.mean(axis=(0, -1))) / std

            std = heartbeat_signal.std(axis=(0, -1))
            std[std == 0] = 1
            heartbeat_signal = (heartbeat_signal - heartbeat_signal.mean(axis=(0, -1))) / std

        # print('window_signal', window_signal.shape)

        if self.nkclean:
            for i in range(window_signal.shape[1]):
                window_signal[:, i] = torch.tensor(nk.ecg_clean(window_signal[:, i].clone().detach().numpy(), sampling_rate=360).copy(), dtype=torch.float32)
                heartbeat_signal[:, i] = torch.tensor(nk.ecg_clean(heartbeat_signal[:, i].clone().detach().numpy(), sampling_rate=360).copy(), dtype=torch.float32)

        window_signal = self.filter_leads(window_signal, header.__dict__['sig_name'])
        heartbeat_signal = self.filter_leads(heartbeat_signal, header.__dict__['sig_name'])

        tortn = {
            'heartbeat': heartbeat_signal,
            'signal': window_signal,
            'label': self.get_label_int(sample['label']),
        }

        if self.use_tab_data:
            comment = self.comments[int(patient)][0].split(' ')
            age = max(0, int(comment[0]))

            is_male = comment[1] == 'M'

            # create pandas row with the tabular data
        
            tab_data = pd.DataFrame({
                'age': [age],
                'is_male': [is_male],
                'RBBB': sample['orig_label'] == 'R',
                'LBBB': sample['orig_label'] == 'L',
                'SB': 'SBR' in sample['extra_annotations'],
                'AF': 'AFIB' in sample['extra_annotations'],
            })
            tortn['tab_data'] = tab_data

        return tortn
    
    def filter_leads(self, signal, leads):
        # convert leads if needed 
        for i, lead in enumerate(leads):
            if lead in conversion.keys():
                leads[i] = conversion[lead]

        # print('leads', leads)

        signal_to_return = torch.zeros(signal.shape[0], len(self.leads_to_use), dtype=torch.float32)
        # if the leads to use are not present in the signal set them to zero
        # leads present in the signal that are not in the leads to use are removed
        # the rest is kept unchanged
        for i, lead in enumerate(self.leads_to_use):
            if lead not in leads:
                signal_to_return[:, i] = torch.zeros(signal.shape[0])
            else:
                signal_to_return[:, i] = signal[:, leads.index(lead)]
        return signal_to_return


    
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

    # pad to same length and pad to match the patch size module
    heartbeat_signals = torch.nn.utils.rnn.pad_sequence(hb, batch_first=True)

    window_signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
    labels = torch.tensor([item['label'] for item in batch])

    tortn = {
        'heartbeat': heartbeat_signals,
        'signal': window_signals,
        'label': labels,
    }

    if 'tab_data' in batch[0].keys():
        tab_data = pd.concat([item['tab_data'] for item in batch], axis=0)
        tortn['tab_data'] = tab_data

    return tortn
