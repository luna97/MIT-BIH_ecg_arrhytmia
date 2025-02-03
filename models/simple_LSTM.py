import torch
import torch.nn as nn

class ECG_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(ECG_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):  # Add lengths for handling variable-length sequences
        # Pack the padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True,enforce_sorted=False)

        # Pass through LSTM
        out_packed, _ = self.lstm(packed_x)  # Output is packed

        # Unpack the output
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

        # Get the output of the last timestep for each sequence (many-to-one)
        out = out[range(len(out)), lengths - 1, :]  # Corrected indexing for lengths


        out = self.fc(out)
        return out


import torch
import torch.nn as nn

class ECG_CONV1D_LSTM(nn.Module):
    def __init__(self, channels, hidden_size, num_layers, num_classes, dropout=0.2, kernel_sizes=[2, 2, 2]):
        super(ECG_CONV1D_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes

        # Added convolutional layers
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=kernel_sizes[0], stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_sizes[1], stride=1)  
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_sizes[2], stride=1)
        self.relu = nn.ReLU() #added activation function
        self.maxpool = nn.MaxPool1d(kernel_size=2) # Added a maxpooling layer


        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, lengths):
        # Reshape input for Conv1d: (batch, channels, seq_len)
        x = x.permute(0, 2, 1) # Set the channels to the second dimension

        # Pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)  
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))


        # Reshape for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)  

        # Adjust lengths for pooling
        #lengths = torch.div(lengths - self.kernel_sizes[0] + 1, 2, rounding_mode='floor')
        #lengths = torch.div(lengths - self.kernel_sizes[1] + 1, 2, rounding_mode='floor')
        #lengths = torch.div(lengths - self.kernel_sizes[2] + 1, 2, rounding_mode='floor')

        # Ensure lengths are at least 1
        #lengths = torch.clamp(lengths, min=1)

        # Pack the padded sequence
        #packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        #out_packed, _ = self.lstm(packed_x)
        out, _ = self.lstm(x)  

        # Unpack the output
        # out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

        # Get the output of the last timestep for each sequence
        # out = out[range(len(out)), lengths - 1, :]
        out = out[:, -1, :]

        out = self.fc(out)
        return out
