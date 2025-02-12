import torch
import os
import pandas as pd
import wfdb

class ECGDataset(torch.utils.data.Dataset):

    def __init__(self, data_folder, subset='train', num_leads=1, name='t_wave_split', oversample=False, random_shift_window=False, patch_size=64, normalize=True):
        self.data_folder = data_folder
        self.subset = subset
        self.samples = []
        self.random_shift_window = random_shift_window
        self.num_leads = num_leads
        self.samples = pd.read_csv(os.path.join(data_folder, name, f'labels_{subset}.csv'))
        if not oversample:
            self.samples = self.samples[self.samples['is_oversampled'] == False]

        self.patch_size = patch_size
        self.normalize = normalize
        
        # get all the different values for column patient
        self.patients = self.samples['patient'].unique()

        # load on memory all the signals
        self.signals = {}
        for patient in self.patients:
            signal, _ = wfdb.rdsamp(os.path.join(data_folder, 'raw', f'{patient}'))
            # normalize the signal
            if normalize:
                signal = (signal - signal.mean(axis=0)) / signal.std(axis=0)
            self.signals[patient] = signal

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
        heartbeat_signal = self.signals[patient][sample['hb_start']:sample['hb_end'], :self.num_leads]
        if self.normalize: heartbeat_signal = (heartbeat_signal - heartbeat_signal.mean(axis=0)) / (heartbeat_signal.std(axis=0) + 1e-6)
        if self.random_shift_window:
            window_start = sample['win_start']
            window_end = sample['win_end']
            shift = torch.randint(- self.patch_size // 3, self.patch_size // 3, (1,)).item() # shift between 0 and patch_size // 3
            window_start += shift
            window_end += shift
            # ensure that the window is inside the signal
            window_start = max(0, window_start)
            window_end = min(window_end, len(self.signals[patient]))
            window_signal = self.signals[patient][window_start:window_end, :self.num_leads]
        else:
            window_signal = self.signals[patient][sample['win_start']:sample['win_end'], :self.num_leads]

        if window_signal.shape[0] == 0:
            print(f'index {idx}, patient {patient}, window_signal shape {window_signal.shape}, heartbeat_signal shape {heartbeat_signal.shape}')

        if self.normalize: window_signal = (window_signal - window_signal.mean(axis=0)) / (window_signal.std(axis=0) + 1e-6)

        return {
            'heartbeat_signal': torch.tensor(heartbeat_signal, dtype=torch.float32),
            'window_signal': torch.tensor(window_signal, dtype=torch.float32),
            'label': self.get_label_int(sample['label'])
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
            # print(f'Adding {num_samples - val_len} samples of patient {patient} to the training set from indexes {count} to {count + num_samples - val_len}')
            # print(f'Adding {val_len} samples of patient {patient} to the validation set from indexes {count + num_samples - val_len} to {count + num_samples}')

            ids_train.extend(range(count, count + num_samples - val_len))
            ids_val.extend(range(count + num_samples - val_len, count + num_samples))
            count += num_samples
        return torch.utils.data.Subset(self, ids_train), torch.utils.data.Subset(self, ids_val)

    
def collate_fn(batch):
    heartbeat_signals = torch.nn.utils.rnn.pad_sequence([item['heartbeat_signal'] for item in batch])
    window_signals = torch.nn.utils.rnn.pad_sequence([item['window_signal'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {
        'heartbeat_signals': heartbeat_signals.squeeze(),
        'window_signals': window_signals.squeeze(),
        'labels': labels
    }
