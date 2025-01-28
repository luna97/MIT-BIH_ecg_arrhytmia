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

