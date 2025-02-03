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

        self.xlstm = get_xlstm(embedding_dim, xlstm_depth=xlstm_depth)
        self.fc = nn.Linear(embedding_dim, num_classes)
        #self.fc = nn.Sequential(
        #    nn.Linear(embedding_dim, embedding_dim // 2),
        #    nn.ReLU(),
        #    nn.Linear(embedding_dim // 2, num_classes),
        #)

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


    def forward(self, x, lengths):
        x = self.convolve(x)
        # print(x.shape)
        x = self.xlstm(x)

        #lengths = self.get_new_lengths(lengths)
        #print('lenghts:', lengths)
        #print('x shape', x.shape)

        # Take the element of x at lengths index for each sample in the batch
        # x = x[torch.arange(x.size(0)), lengths - 1]
        #print('x shape', x.shape)
        x = x[:, -1, :]
        x = self.fc(x)

        return x
    
    def get_new_lengths(self, lengths):
        new_lengths = (lengths + 2 * self.conv1d1a.padding[0] - self.conv1d1a.dilation[0] * (self.conv1d1a.kernel_size[0] - 1) - 1) // self.conv1d1a.stride[0] + 1
        new_lengths = (new_lengths + 2 * self.maxpool.padding - (self.maxpool.kernel_size - 1) - 1) // self.maxpool.stride + 1
        new_lengths = (new_lengths + 2 * self.conv1d2.padding[0] - (self.conv1d2.kernel_size[0] - 1) - 1) // self.conv1d2.stride[0] + 1
        new_lengths = (new_lengths + 2 * self.maxpool.padding - (self.maxpool.kernel_size - 1) - 1) // self.maxpool.stride + 1
        new_lengths = (new_lengths + 2 * self.conv1d3.padding[0] - (self.conv1d3.kernel_size[0] - 1) - 1) // self.conv1d3.stride[0] + 1
        new_lengths = (new_lengths + 2 * self.maxpool.padding - (self.maxpool.kernel_size - 1) - 1) // self.maxpool.stride + 1
        return new_lengths

    
    def trainable_parameters(self):
        return self.parameters()


def get_xlstm(embedding_dim, xlstm_depth=1):
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
        dropout=0.2,
        slstm_at=range(0, xlstm_depth),
    )

    return xLSTMBlockStack(cfg)