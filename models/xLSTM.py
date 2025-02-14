import torch
from torch import nn

from models.utils import get_activation_fn, get_xlstm
from models.PatchEmbedding import PatchEmbedding 

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

        self.patch_embedding = PatchEmbedding(patch_size=patch_size, num_hiddens=embedding_dim)

        self.sep_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.down_project = nn.LazyLinear(out_features=embedding_dim)

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
    

    def reconstruct(self, x):
        x = x.permute(0, 2, 1) # put the channels in the middle
        x = self.patch_embedding(x)
        # print('x shape post patch_emb', x.shape)
        x = self.xlstm_forward(x)
        x = self.reconstruction(x)
        # print('x shape post reconstruction', x.shape) # [batch_size, seq_len, embedding_dim]
        x = x.view(x.shape[0], -1) 
        # print('x shape post view', x.shape) # [batch_size, seq_len * embedding_dim]
        return x

    def xlstm_forward(self, x):
        # [bs, seq_len, embedding_dim]
        x1 = self.xlstm(x)
        # print('x shape', x.shape)
        # x2 = self.xlstm_bi(x.flip(dims=(1, 2))).flip(dims=(1, 2))
        return x1 # + x2


    def forward(self, ctx, x):
        ctx = self.patch_embedding(ctx)
        x = self.patch_embedding(x)
        # add the separation token between the context and the input
        sep_token = self.sep_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([ctx, sep_token, x], dim=1)
        x = self.xlstm_forward(x)[:, -1, :] # [batch_size, embedding_dim]
        x = self.fc(x)
        return x

    def trainable_parameters(self):
        return self.parameters()
    
    def head_parameters(self):
        return self.fc.parameters() 
    