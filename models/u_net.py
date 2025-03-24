
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.utils as mu


class DoubleXLSTMDown(nn.Module):
    def __init__(self, emb_size, final_size=None, dropout=0.1):
        super(DoubleXLSTMDown, self).__init__()
        self.xlstm = mu.get_large_xlstm(emb_size, dropout=dropout, blocks=['m', 'm'])
        self.final_size = final_size
        if final_size is not None:
            self.final_layer = nn.Linear(emb_size, final_size)

    def forward(self, x):
        # print('x_before', x.shape)
        x = self.xlstm(x)
        if self.final_size is not None:
            x = self.final_layer(x)
        # print('x_after', x.shape)
        return x
    
class DoubleXLSTMUp(nn.Module):
    def __init__(self, emb_size, initial_size=None, dropout=0.1):
        super(DoubleXLSTMUp, self).__init__()
        self.xlstm = mu.get_large_xlstm(emb_size, dropout=dropout, blocks=['m', 'm'])
        self.initial_size = initial_size
        if initial_size is not None:
            self.initial_layer = nn.Linear(initial_size, emb_size)

    def forward(self, x1, x2):
        if self.initial_size is not None:
            
            x = self.initial_layer(x1)
        x = x + x2
        x = self.xlstm(x)
        return x