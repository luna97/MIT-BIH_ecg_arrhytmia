import torch
import numpy as np
import random
from torch import nn
from xlstm import FeedForwardConfig, mLSTMLayerConfig, mLSTMBlockConfig, sLSTMLayerConfig, sLSTMBlockConfig, xLSTMBlockStackConfig, xLSTMBlockStack
from xlstm.xlstm_large import xLSTMLargeConfig
from xlstm.xlstm_large.model import xLSTMLargeBlockStack
from models.modules import mLSTMWrapper, LinearPatchEmbedding, ConvPatchEmbedding, ONNConvPatchEmbedding, UNetPatchEmbedding, HeadModule, EmbedPatching, UNetEmbedPatching
import os

def get_patch_embedding(type, patch_size, num_hiddens, num_channels):
    if type == 'linear':
        print('using linear patch embedding')
        return LinearPatchEmbedding(patch_size=patch_size, num_hiddens=num_hiddens)
    if type == 'conv':
        print('using conv patch embedding')
        return ConvPatchEmbedding(patch_size=patch_size, num_hiddens=num_hiddens, num_channels=num_channels)
    if type == 'onn':
        print('using ONN patch embedding')
        return ONNConvPatchEmbedding(patch_size=patch_size, num_hiddens=num_hiddens, num_channels=num_channels)
    if type == 'unet':
        print('using UNet patch embedding')
        return UNetPatchEmbedding(patch_size=patch_size, num_hiddens=num_hiddens, num_channels=num_channels)

def get_reconstruction_head(type, patch_size, embedding_size, num_channels, activation_fn):
    if type == 'linear':
        return EmbedPatching(
            patch_size=patch_size, 
            num_hiddens=embedding_size, 
            num_channels=num_channels, 
            activation_fn=activation_fn, 
            use_pre_head=True
        )
    if type == 'unet':
        return UNetEmbedPatching(
            patch_size=patch_size, 
            num_hiddens=embedding_size, 
            num_channels=num_channels
        )

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


def get_xlstm(
        embedding_dim, 
        dropout=0.2, 
        blocks=['m', 's', 'm', 'm', 'm', 'm', 'm'],
        num_heads=4
    ):
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, 
                qkv_proj_blocksize=num_heads, 
                num_heads=num_heads
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=num_heads,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),
        context_length=700,
        num_blocks=len(blocks),
        embedding_dim=embedding_dim,
        slstm_at=[1 if b == 's' else 0 for b in blocks],
        dropout=dropout,
    )

    return xLSTMBlockStack(cfg)


def get_large_xlstm(       
        embedding_dim, 
        dropout=0.2, 
        blocks=['m', 'm', 'm', 'm', 'm', 'm', 'm'],
        num_heads=4,
    ):
    xlstm_config = xLSTMLargeConfig(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_blocks=len(blocks),
        vocab_size=0,
        return_last_states=True,
        mode="train",
        chunkwise_kernel="chunkwise--triton_xl_chunk", # xl_chunk == TFLA kernels
        sequence_kernel="native_sequence__triton",
        step_kernel="triton",
    )

    blocks = xLSTMLargeBlockStack(xlstm_config)
    return mLSTMWrapper(blocks, dropout=dropout)
