import torch
from torch import nn

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class myxLSTM(nn.Module):

    def __init__(self, in_channels, dropout=0.2, num_classes=5, xlstm_depth=1, activation_fn='leakyrelu', pooling='max', num_leads=2):
        super(myxLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)


        self.out_ch_1 = 16
        self.conv1d1a = nn.Conv1d(in_channels, self.out_ch_1, kernel_size=4, padding=1, bias=False)
        self.conv1d1b = nn.Conv1d(in_channels, self.out_ch_1, kernel_size=2, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(self.out_ch_1 * 2)

        self.out_ch_2 = 64
        self.conv1d2 = nn.Conv1d(self.out_ch_1 * 2, self.out_ch_2, kernel_size=3, padding=0, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(self.out_ch_2)

        self.out_ch_3 = 128
        self.conv1d3 = nn.Conv1d(self.out_ch_2, self.out_ch_3, kernel_size=3, padding=0, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(self.out_ch_3)

        self.pool_size = 2
        if pooling == 'max':
            self.maxpool = nn.MaxPool1d(kernel_size=self.pool_size)
        elif pooling == 'avg':
            self.maxpool = nn.AvgPool1d(kernel_size=self.pool_size)
        else:
            raise ValueError(f"Pooling method {pooling} not supported")
        
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Activation function {activation_fn} not supported")

        # dummy sample
        x = torch.randn(1, 200, num_leads)
        x = self.convolve(x)
        embedding_dim = x.shape[-1]
        print(f"Embedding Dim: {x.shape}")

        self.xlstm = get_xlstm(embedding_dim, xlstm_depth=xlstm_depth, dropout=dropout)

        #self.fc = nn.Linear(embedding_dim, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            self.activation,
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def convolve(self, x):
        x = x.permute(0, 2, 1) # move channels to the last dimension

        # first conv block
        x1 = self.conv1d1a(x)
        x2 = self.conv1d1b(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        # second conv block
        x = self.conv1d2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        # third conv block
        x = self.conv1d3(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = x.permute(0, 2, 1)
        return x
    

    def get_embeddings(self, x):
        x = self.convolve(x)
        x = self.xlstm(x)
        return x[:, -1, :]


    def forward(self, x, lengths=None):
        x = self.get_embeddings(x)
        x = self.fc(x)
        return x


    def trainable_parameters(self):
        return self.parameters()


def get_xlstm(embedding_dim, xlstm_depth=1, dropout=0.2):
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=3, 
                qkv_proj_blocksize=4, 
                num_heads=4,
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=4,
                conv1d_kernel_size=3,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),
        num_blocks=xlstm_depth,
        embedding_dim=embedding_dim,
        dropout=dropout,
        context_length=700,
        slstm_at=range(0, xlstm_depth, 2),
    )

    return xLSTMBlockStack(cfg)