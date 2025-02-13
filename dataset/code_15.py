from torch.utils.data import Dataset, random_split
import torch
import numpy as np
import wfdb
import os
from dataset.helper_code_code15 import find_records, load_label, load_signals
from neurokit2 import signal_resample

class ECGCODE15Dataset(Dataset):
    def __init__(self, data_folder, normalize=True, num_leads=1, random_shift=False, patch_size=64):
        """
        Args:
            records (list): List of records of ECG traces
        """
        self.records = find_records(data_folder)
        print(f'loaded {len(self.records)} records')
        self.data_folder = data_folder
        self.normalize = normalize 
        self.num_leads = num_leads
        self.random_shift = random_shift
        self.patch_size = patch_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        
        signal, _ = load_signals(self.data_folder + '/' + record)

        # remove a random number of datapoints from the signal from 0 to patch size 
        if self.random_shift:
            shift = np.random.randint(0, self.patch_size // 2)
            signal = signal[shift:]

        signal = torch.tensor(signal[:, :self.num_leads], dtype=torch.float32)

        # normalize the signal by subtracting the mean and dividing by the standard deviation
        if self.normalize: signal = (signal - signal.mean(axis=0)) / signal.std(axis=0)


        # print(f'idx {idx}, signal shape {signal.shape}')

        return {
            'signal': signal
            # 'label': torch.tensor(label, dtype=torch.float)
        }
    
    def split_validation_training(self, val_size_pct = 0.1):
        """
        Split the dataset into training and validation sets.
        
        Args:
            val_size_pct (float): Percentage of the dataset to include in the validation set
        """
        dataset_size = len(self)
        train_size = int(dataset_size * (1 - val_size_pct))
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size])
        return train_dataset, val_dataset
    
def collate_fn(batch):
    signals = torch.nn.utils.rnn.pad_sequence([item['signal'] for item in batch])
    # print(f'signals shape {signals.shape}')
    return {
        'signal': signals.squeeze(),
    }

