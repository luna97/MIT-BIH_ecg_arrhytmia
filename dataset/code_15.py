from torch.utils.data import Dataset, random_split
import torch
import numpy as np
import wfdb
import os
import pandas as pd
from dataset.helper_code_code15 import find_records, load_label, load_signals

class ECGCODE15Dataset(Dataset):
    def __init__(self, config, num_leads=1):
        """
        Args:
            records (list): List of records of ECG traces
        """
        self.records = find_records(config.data_folder_code15)
        print(f'loaded {len(self.records)} records')
        self.data_folder = config.data_folder_code15
        self.normalize = config.normalize 
        self.num_leads = num_leads
        self.random_shift = config.random_shift
        self.patch_size = config.patch_size
        self.use_tab_data = config.use_tab_data
        
        if self.use_tab_data:
            self.load_tabular_data()

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.data_folder + '/../exams.csv')
        # set exam_id as index
        self.tab_data.set_index('exam_id', inplace=True)
        # remove trace_file, patient_id and nn_predicted_age
        self.tab_data.drop(columns=['trace_file', 'patient_id', 'nn_predicted_age', 'death', 'timey'], inplace=True)
        print("tabular data fields for CODE 15: ", self.tab_data.columns)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        
        signal, _ = load_signals(self.data_folder + '/' + record)

        # remove a random number of datapoints from the signal from 0 to patch size 
        if self.random_shift:
            shift = np.random.randint(0, self.patch_size // 2)
            if shift > 0 and len(signal) > shift:
                signal = signal[shift:]

        signal = torch.tensor(signal[:, :self.num_leads], dtype=torch.float32)
        # print('dioss shape', signal.shape)

        # normalize the signal by subtracting the mean and dividing by the standard deviation
        if self.normalize:
            std = signal.std(axis=0)
            if std != 0:
                signal = (signal - signal.mean(axis=0)) / std

        tortn = {
            'signal': signal,
        }

        if self.use_tab_data:
            tab_data = self.tab_data.loc[int(record)]
            # to dataframe
            tab_data = pd.DataFrame(tab_data)
            tortn['tab_data'] = tab_data
        
        return tortn
        
    
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
    signals = [item['signal'] for item in batch]
    masks = [torch.ones_like(item['signal'], dtype=torch.bool) for item in batch]
    try:
        padded_signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
    except:
        print('signals', signals)
        print('signals shape', [s.shape for s in signals])
        
    padded_masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)

    tortn =  {
        'signal': padded_signals,
        'mask': padded_masks,
    }

    if 'tab_data' in batch[0].keys():
        tab_data = pd.concat([item['tab_data'] for item in batch], axis=0)
        tortn['tab_data'] = tab_data

    return tortn

