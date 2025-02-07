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

    def __init__(self, in_channels, dropout=0.2, num_classes=5, xlstm_depth=1, activation_fn='leakyrelu', pooling='max', num_leads=2, channels=[32, 64, 128]):
        super(myxLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_ch = in_channels
        for out_ch in channels:
            self.convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch

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

        self.slstm1a = get_mLSTM(embedding_dim, dropout=dropout)
        self.slstm1b = get_mLSTM(embedding_dim, dropout=dropout)

        self.mlstma = get_sLSTM(embedding_dim, dropout=dropout)
        self.mlstmb = get_sLSTM(embedding_dim, dropout=dropout)

        #self.fc = nn.Linear(embedding_dim, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            self.activation,
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def convolve(self, x):
        x = x.permute(0, 2, 1)  # move channels to the last dimension

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.maxpool(x)

        x = x.permute(0, 2, 1)
        return x
            

    def get_embeddings(self, x):
        x = self.convolve(x)
        x1 = self.slstm1a(x)
        x2 = self.slstm1b(x.flip(1))
        x = x1 + x2.flip(1)
        x1 = self.mlstma(x)
        x2 = self.mlstmb(x.flip(1))
        x = x1 + x2.flip(1)
        return x[:, -1, :]


    def forward(self, x, lengths=None):
        x = self.get_embeddings(x)
        x = self.fc(x)
        return x


    def trainable_parameters(self):
        return self.parameters()
    
def get_sLSTM(embedding_dim, dropout=0.2):
    cfg = xLSTMBlockStackConfig(
        mlstm_block=None,
        slstm_block = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=4,
                conv1d_kernel_size=3,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),
        num_blocks=1,
        embedding_dim=embedding_dim,
        dropout=dropout,
        slstm_at=[0],
    )

    return xLSTMBlockStack(cfg)

def get_mLSTM(embedding_dim, dropout=0.2):
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=3, 
                qkv_proj_blocksize=4, 
                num_heads=4,
            )
        ),
        slstm_block=None,
        num_blocks=1,
        embedding_dim=embedding_dim,
        dropout=dropout,
        context_length=700,
        slstm_at=[],
    )

    return xLSTMBlockStack(cfg)


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