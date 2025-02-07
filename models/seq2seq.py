import torch
import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, char2numY, n_channels=10, input_depth=280, num_units=128, max_time=10, bidirectional=False):
        super(Seq2SeqModel, self).__init__()
        self.n_channels = n_channels
        self.max_time = max_time
        self.bidirectional = bidirectional
        self.num_units = num_units if not bidirectional else 2* num_units

        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=2, stride=1, padding='same')
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)  # 'same' padding not available

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding='same')
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding='same')


        self.embed_size = 10
        self.dec_embedding = nn.Embedding(len(char2numY), self.embed_size)

        self.lstm_enc = nn.LSTM(input_size=self.conv3.out_channels * (input_depth // (self.n_channels*4)), #adjusted input size due to maxpooling halving the size twice
                            hidden_size=num_units,
                                 bidirectional=bidirectional, batch_first = True)


        self.lstm_dec = nn.LSTM(input_size =self.embed_size, 
                                hidden_size=self.num_units,
                                batch_first=True)

        self.dense = nn.Linear(self.num_units, len(char2numY))


    def forward(self, inputs, dec_inputs):
        # Reshape inputs
        _inputs = inputs.view(-1, self.n_channels, inputs.shape[2] // self.n_channels)

        # Convolutional layers
        conv1_out = torch.relu(self.conv1(_inputs))
        max_pool_1_out = self.max_pool_1(conv1_out)

        conv2_out = torch.relu(self.conv2(max_pool_1_out))
        max_pool_2_out = self.max_pool_2(conv2_out)

        conv3_out = torch.relu(self.conv3(max_pool_2_out))
        

        # Reshape for LSTM input
        data_input_embed = conv3_out.view(-1, self.max_time, conv3_out.shape[1] * conv3_out.shape[2])
       
        # Decoder embedding
        data_output_embed = self.dec_embedding(dec_inputs)


        # Encoder LSTM
        _, (hidden, cell)  = self.lstm_enc(data_input_embed)
        if self.bidirectional:
            hidden = torch.cat((hidden[0], hidden[1]), dim=1)
            cell = torch.cat((cell[0], cell[1]), dim=1)


        # Decoder LSTM
        dec_outputs, _ = self.lstm_dec(data_output_embed, (hidden, cell))


        # Dense layer
        logits = self.dense(dec_outputs)

        return logits