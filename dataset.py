import os
import torch

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, subset='train'):
        self.data_folder = data_folder
        self.subset = subset
        self.samples = []
        with open(os.path.join(data_folder, subset, 'labels.csv'), 'r') as f:
            for line in f:
                sample, label = line.strip().split(',')
                self.samples.append((sample, label))

    def __len__(self):
        return len(self.samples)
    
    def get_label_int(self, label):
        if label == 'N': return 0
        if label == 'S': return 1
        if label == 'V': return 2
        if label == 'F': return 3
        if label == 'Q': return 4

    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        signal = torch.load(os.path.join(self.data_folder, self.subset, f'{sample}.pt'), weights_only=True)
        return signal, self.get_label_int(label)
    

def collate_fn(batch):
    signals, labels = zip(*batch)
    lengths = torch.tensor([len(sig) for sig in signals])  # Calculate lengths
    signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True, padding_value=0) #Explicitly set padding to 0
    labels = torch.tensor([int(label) for label in labels])
    return signals, labels, lengths # Return lengths