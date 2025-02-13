import torch
from torch import nn

from fastonn import SelfONN1d

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

    def __init__(
            self, 
            dropout=0.2, 
            num_classes=5, 
            xlstm_depth=1, 
            activation_fn='leakyrelu', 
            pooling='max', 
            embedding_dim=64, 
            patch_size=64
        ): 
        super(myxLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.patch_size = patch_size

        self.activation = get_activation_fn(activation_fn)
    
        
        if pooling == 'max':
            self.pool = nn.MaxPool1d(kernel_size=2)
        elif pooling == 'avg':
            self.pool = nn.AvgPool1d(kernel_size=2)

        size_ln1 = (self.patch_size - 4) 
        size_ln2 = (size_ln1 - 4) 
        size_ln3 = (size_ln2 - 4) 

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=0), # from 64 to 6
            # self.pool, # from 60 to 30, 11 to 7
            self.activation,
            nn.LayerNorm(size_ln1),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=0), # from 30 to 26, 7 to 5
            # self.pool, # from 26 to 13, 5 to 2
            self.activation,
            nn.LayerNorm(size_ln2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, padding=0), # from 13 to 9
            # self.pool, # from 9 to 4
            self.activation,
            nn.LayerNorm(size_ln3),
            nn.Dropout(dropout),
        )

        self.sep_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        sample = torch.randn(1, 1, patch_size)
        after_convs = self.encoder(sample)
        flattened_size = after_convs.view(1, -1).shape[1]
        print('flattened_size', flattened_size)

        self.down_project = nn.Linear(flattened_size, embedding_dim)

        self.xlstm = get_xlstm(embedding_dim, xlstm_depth, dropout=dropout)
        # self.xlstm_bi = get_xlstm(embedding_dim, xlstm_depth, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            self.activation,
            nn.Linear(embedding_dim // 2, num_classes),
        )

        self.reconstruction = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            self.activation,
            nn.Linear(embedding_dim // 2, self.patch_size),
        )

    def seq_to_token(self, x):
        # convert from [batch_size, seq_len, 1] to [batch_size, seq_len // 64, 64], padding with zeros if necessary
        x = x.permute(1, 0) 
        batch_size, seq_len = x.shape
        # print('starting from a tensor of shape', x.shape)
        if seq_len % self.patch_size != 0:
            x = x[:, :seq_len - (seq_len % self.patch_size)]
        
        num_patches = x.shape[1] // self.patch_size
        # print('num_patches', num_patches)
        x = x.view(batch_size, -1, 1, self.patch_size)
        # print('dim after seq to token', x.shape)
        return x, batch_size, num_patches
    
    def tokenize_signal(self, x):
        if len(x.shape) == 1:
            # it may miss the batch dimension for the case of a single sample
            x = x.unsqueeze(-1)
        x, batch_size, num_patches = self.seq_to_token(x)
        x = x.reshape(x.shape[0] * x.shape[1], 1, self.patch_size)
        x = self.encoder(x) # [batch_size, seq_len // 64, 128, 4]
        x = x.view(batch_size, num_patches, -1)
        x = self.down_project(x) # [batch_size, seq_len // 64, embedding_dim]
        return x
    

    def reconstruct(self, x):
        x = self.tokenize_signal(x)
        x = self.xlstm_forward(x)
        x = self.reconstruction(x)
        return x

    def xlstm_forward(self, x):
        # [bs, seq_len, embedding_dim]
        x1 = self.xlstm(x)
        # print('x shape', x.shape)
        # x2 = self.xlstm_bi(x.flip(dims=(1, 2))).flip(dims=(1, 2))
        return x1 # + x2


    def forward(self, ctx, x):
        ctx = self.tokenize_signal(ctx)
        x = self.tokenize_signal(x)
        # add the separation token between the context and the input
        sep_token = self.sep_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([ctx, sep_token, x], dim=1)
        x = self.xlstm_forward(x)[:, -1, :] # [batch_size, embedding_dim]
        x = self.fc(x)
        return x

    def trainable_parameters(self):
        return self.parameters()
    

def get_activation_fn(activation_fn):
    if activation_fn == 'relu':
        return nn.ReLU()
    elif activation_fn == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation_fn == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Activation function {activation_fn} not supported")
    
def get_pooling(pooling, kernel_size=2):
    if pooling == 'max':
        return nn.MaxPool1d(kernel_size=kernel_size)
    elif pooling == 'avg':
        return nn.AvgPool1d(kernel_size=kernel_size)
    else:
        raise ValueError(f"Pooling {pooling} not supported")
    
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
        context_length=500,
        slstm_at=[0] if xlstm_depth == 1 else [1],
    )

    return xLSTMBlockStack(cfg)