import torch
import os
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np
from dataset.generic_utils import random_shift, find_records
from torch.utils.data import random_split
from joblib import Parallel, delayed

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class ECGMIMICDataset(torch.utils.data.Dataset):

    def __init__(self, config, leads_to_use=leads):
        self.data_folder = config.data_folder_mimic
        self.random_shift = config.random_shift
        self.nkclean = config.nk_clean
        self.leads = leads if leads_to_use == ['*'] else leads_to_use
        self.use_tab_data = config.use_tab_data
        self.patch_size = config.patch_size
        self.normalize = config.normalize
        self.labels_file = config.labels_file_mimic
        self.records = find_records(config.data_folder_mimic)
        print(f'loaded {len(self.records)} records') 
        if self.use_tab_data:
            self.load_tabular_data()

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        # set exam_id as index
        self.tab_data.set_index('exam_id', inplace=True)
        # remove trace_file, patient_id and nn_predicted_age
        self.tab_data.drop(columns=['trace_file', 'patient_id', 'nn_predicted_age', 'death', 'timey'], inplace=True)
        print("tabular data fields for CODE 15: ", self.tab_data.columns)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        signal, _ = wfdb.rdsamp(os.path.join(self.data_folder, record))
        
        if self.random_shift: signal = random_shift(signal, self.patch_size)

        signal = torch.tensor(signal, dtype=torch.float32)

        if self.leads != leads:
            # keep the selected leads
            signal = signal[:, [leads.index(lead) for lead in self.leads]].squeeze()

        # normalize the signal by subtracting the mean and dividing by the standard deviation
        if self.normalize:
            std = signal.std(axis=(0, -1))
            std[std == 0] = 1 # avoid division by zero, samples with std = 0 are all zero
            signal = (signal - signal.mean(axis=(0, -1))) / std

        return {
            'signal':signal,
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
    signals = [item['signal'] for item in batch]
    padded_signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)

    tortn =  {
        'signal': padded_signals,
    }

    return tortn
