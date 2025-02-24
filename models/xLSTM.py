import torch
from torch import nn

from models.utils import get_activation_fn, get_xlstm
from models.PatchEmbedding import TabularEmbeddings, PatchEmbedding, FeatureSpec

class myxLSTM(nn.Module):

    def __init__(
            self, 
            num_classes=5,
            patch_size=64,
            dropout=0.3,
            multi_token_prediction=True,
            activation_fn='leakyrelu',
            embedding_size=1024,
            xlstm_depth=8,
        ): 
        super(myxLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.patch_size = patch_size
        self.multi_token_prediction = multi_token_prediction

        self.activation = get_activation_fn(activation_fn)

        self.patch_embedding = PatchEmbedding(patch_size=patch_size, num_hiddens=embedding_size)

        self.sep_token = nn.Parameter(torch.randn(1, 1, embedding_size))

        self.xlstm = get_xlstm(embedding_size, xlstm_depth, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            self.activation,
            nn.Linear(embedding_size // 2, num_classes),
        )

        if not self.multi_token_prediction:
            self.reconstruction = HeadModule(embedding_size, patch_size)
        else:
            self.rec1 = HeadModule(embedding_size, patch_size, dropout)
            self.rec2 = HeadModule(embedding_size, patch_size, dropout)
            self.rec3 = HeadModule(embedding_size, patch_size, dropout)
            self.rec4 = HeadModule(embedding_size, patch_size, dropout)
    
        self.tab_embeddings = TabularEmbeddings([
            FeatureSpec('age', 110, torch.int64),
            FeatureSpec('is_male', 2, torch.bool),
            FeatureSpec('RBBB', 2, torch.bool),
            FeatureSpec('LBBB', 2, torch.bool),
            # FeatureSpec('normal_ecg', 2, torch.bool),
        ], num_hiddens=embedding_size, dropout=dropout)


    def reconstruct(self, x, tab_data):
        batch_size = x.shape[0]
        tab_emb = self.tab_embeddings(tab_data, batch_size)
        _, num_embeddings, _ = tab_emb.shape
        x = x.permute(0, 2, 1) # put the channels in the middle
        x = self.patch_embedding(x)

        x = torch.cat([tab_emb, x], dim=1)
        x = self.xlstm(x)

        x = x[:, num_embeddings:, :]

        if self.multi_token_prediction:
            x1 = self.rec1(x)
            x2 = self.rec2(x)
            x3 = self.rec3(x)
            x4 = self.rec4(x)
            return x1, x2, x3, x4
        else:
            x = self.reconstruction(x)
            return x
    

    def forward(self, ctx, x, tab_data):
        x = x.permute(0, 2, 1) # put the channels in the middle
        ctx = ctx.permute(0, 2, 1) # put the channels in the middle
        ctx = self.patch_embedding(ctx)
        x = self.patch_embedding(x)
        # add the separation token between the context and the input
        sep_token = self.sep_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([ctx, sep_token, x], dim=1)
        x = self.xlstm(x)[:, -1, :] # [batch_size, embedding_dim]
        x = self.fc(x)
        return x

    def trainable_parameters(self):
        return self.parameters()
    
    def head_parameters(self):
        return self.fc.parameters() 
    

class HeadModule(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super(HeadModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, out_features),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], -1) 
        return x