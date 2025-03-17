import torch
import os
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np
from dataset.generic_utils import check_mean_var_r_peaks

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
conversion = {
    'MLII' : 'II',
}

class ECGMITBIHDataset(torch.utils.data.Dataset):
    def __init__(self, config, subset='train', name='t_wave_split', use_labels_in_tab_data=True, random_shift=False):
        """
        Args:
            config: configuration object
            subset: 'train' or 'test'
            name: name of the labels folder, default is 't_wave_split' so here the in the labels csv file the heartbeats will be divided by the t wave
            use_labels_in_tab_data: if True the labels are used in the tabular data (like the rbbb, lbbb, etc)
        """
        
        self.data_folder = config.data_folder_mit
        self.subset = subset
        self.samples = []
        self.random_shift = random_shift
        self.nkclean = config.nk_clean
        self.use_tab_data = config.use_tab_data
        self.patch_size = config.patch_size
        self.normalize = config.normalize

        self.leads_to_use = leads if config.leads == ['*'] else config.leads
        self.use_labels_in_tab_data = use_labels_in_tab_data

        self.samples = pd.read_csv(os.path.join(self.data_folder, name, f'labels_{subset}.csv'))
        # ensure no Nan values
        self.samples['extra_annotations'] = self.samples['extra_annotations'].fillna('')

        if not config.oversample:
            self.samples = self.samples[self.samples['is_oversampled'] == False]
        
        print(self.samples.head())  
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
        else:
            window_start = sample['win_start']
            window_end = sample['win_end']

        window_signal = signal[window_start:window_end]

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
            'r_peak_interval_mean': torch.tensor(sample['r_peaks_interval_mean']),
            'r_peak_variance' : torch.tensor(sample['r_peak_variance']),
            'patient_id': patient,
        }

        tortn = check_mean_var_r_peaks(tortn)

        # ensure no nan
        if torch.isnan(tortn['r_peak_interval_mean']):
            tortn['r_peak_interval_mean'] = torch.tensor(0)
        if torch.isnan(tortn['r_peak_variance']):
            tortn['r_peak_variance'] = torch.tensor(0)  

        if self.use_tab_data:
            comment = header.__dict__['comments'][0].split(' ')
            age = max(0, int(comment[0]))
            is_male = comment[1] == 'M'

            # create pandas row with the tabular data
            tab_data = {
                'age': [age],
                'is_male': [is_male]
            }
            if self.use_labels_in_tab_data:
                tab_data['RBBB'] = sample['orig_label'] == 'R'
                tab_data['LBBB'] = sample['orig_label'] == 'L'
                tab_data['SB'] = 'SBR' in sample['extra_annotations'] # sinus bradycardia
                tab_data['ST'] = False
                tab_data['AF'] = 'AFIB' in sample['extra_annotations'] # atrial fibrillation

            tortn['tab_data'] = pd.DataFrame(tab_data)

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


    
    def split_validation_training(self, val_size=0.2, split_by_patient=False):
        train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
        val = [203, 114]
        #       N     S  V  F  Q
        # 101: [1860, 3, 0, 0, 2], 
        # 106: [1507, 0, 520, 0, 0], 
        # 108: [1740, 4, 17, 2, 0], 
        # 109: [2492, 0, 38, 2, 0], 
        # 112: [2537, 2, 0, 0, 0], 
        # 114: [1820, 12, 43, 4, 0], <-----
        # 115: [1953, 0, 0, 0, 0], 
        # 116: [2302, 1, 109, 0, 0], 
        # 118: [2166, 96, 16, 0, 0], 
        # 119: [1543, 0, 444, 0, 0], 
        # 122: [2476, 0, 0, 0, 0], 
        # 124: [1536, 31, 47, 5, 0],
        # 201: [1635, 128, 198, 2, 0],
        # 203: [2529, 2, 444, 1, 4], <-----
        # 205: [2571, 3, 71, 11, 0], 
        # 207: [1543, 107, 210, 0, 0], 
        # 208: [1586, 2, 992, 373, 2], 
        # 209: [2621, 383, 1, 0, 0], 
        # 215: [3195, 3, 164, 1, 0], 
        # 220: [1954, 94, 0, 0, 0],
        # 223: [2045, 73, 473, 14, 0], 
        # 230: [2255, 0, 1, 0, 0]}

        if self.subset != 'train':
            raise ValueError('Can only split the training dataset')

        count = 0
        ids_val, ids_train = [], []
        for patient in self.patients:
            df_patient = self.samples[self.samples['patient'] == patient]
            num_samples = len(df_patient)

            if split_by_patient:
                if patient in val:
                    ids_val.extend(range(count, count + num_samples))
                else:
                    ids_train.extend(range(count, count + num_samples))
            else:
                val_len = int(val_size * num_samples)

                # the first 0.8 of the patients go to the training set
                ids_train.extend(range(count, count + num_samples - val_len))
                ids_val.extend(range(count + num_samples - val_len, count + num_samples))
            count += num_samples
        return torch.utils.data.Subset(self, ids_train), torch.utils.data.Subset(self, ids_val)

    
def collate_fn(batch):
    signals = [item['signal'] for item in batch]
    hb = [item['heartbeat'] for item in batch]
    patients = [item['patient_id'] for item in batch]

    r_peak_interval_mean = torch.tensor([item['r_peak_interval_mean'] for item in batch])
    r_peak_variance = torch.tensor([item['r_peak_variance'] for item in batch])

    # pad to same length and pad to match the patch size module
    heartbeat_signals = torch.nn.utils.rnn.pad_sequence(hb, batch_first=True)

    window_signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
    labels = torch.tensor([item['label'] for item in batch])

    tortn = {
        'heartbeat': heartbeat_signals,
        'signal': window_signals,
        'label': labels,
        'r_peak_interval_mean': r_peak_interval_mean,
        'r_peak_variance': r_peak_variance,
        'patient_ids': torch.tensor(patients),
    }

    if 'tab_data' in batch[0].keys():
        tab_data = pd.concat([item['tab_data'] for item in batch], axis=0)
        tortn['tab_data'] = tab_data

    return tortn
