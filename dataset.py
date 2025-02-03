import os
import torch

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, subset='train', transform=None, classes=['N', 'S', 'V', 'F', 'Q'], num_leads=2):
        self.data_folder = data_folder
        self.subset = subset
        self.samples = []
        self.transform = transform
        self.num_leads = num_leads
        with open(os.path.join(data_folder, subset, 'labels.csv'), 'r') as f:
            for line in f:
                sample, label = line.strip().split(',')
                if label in classes:
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
        # needed for contrastive learning
        patient_id = sample.split('_')[0]
        signal = torch.load(os.path.join(self.data_folder, self.subset, f'{sample}.pt'), weights_only=True)
       
        if self.transform: # Check if transform is provided and apply it
            signal = self.transform(signal)

        signal = signal[..., :self.num_leads]
            
        return signal, self.get_label_int(label), patient_id
    
    def get_labels(self):
        return [self.get_label_int(label) for _, label in self.samples]
    

def collate_fn(batch):
    signals, labels, patients_ids = zip(*batch)
    # from tuple to list
    patients_ids = torch.tensor([int(p) for p in list(patients_ids)])
    lengths = torch.tensor([len(sig) for sig in signals])  # Calculate lengths
    signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True, padding_value=0) #Explicitly set padding to 0
    labels = torch.tensor([int(label) for label in labels])
    return signals, labels, lengths, patients_ids # Return lengths