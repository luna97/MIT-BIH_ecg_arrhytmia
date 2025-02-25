import torch
import numpy as np
import random
from torch import nn
from xlstm import FeedForwardConfig, mLSTMLayerConfig, mLSTMBlockConfig, sLSTMLayerConfig, sLSTMBlockConfig, xLSTMBlockStackConfig, xLSTMBlockStack
from models.SeriesDecomposition import SeriesDecomposition

def get_activation_fn(activation_fn):
    if activation_fn == 'relu':
        return nn.ReLU()
    elif activation_fn == 'leakyrelu' or activation_fn == 'leaky_relu':
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


def get_xlstm(
        embedding_dim, 
        dropout=0.2, 
        blocks=['m', 's', 'm', 'm', 'm', 'm', 'm']
    ):
    xlstm = nn.ModuleList()
    for block in blocks:
        if block == 'm':
            xlstm.append(get_mLSTM(embedding_dim, dropout)) 
        elif block == 's':
            xlstm.append(get_sLSTM(embedding_dim, dropout))
        else:
            raise ValueError(f"Block {block} not supported")
        
    return nn.Sequential(*xlstm)