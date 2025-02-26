import torch
from torch import nn

from models.utils import get_activation_fn, get_xlstm
from models.PatchEmbedding import TabularEmbeddings, PatchEmbedding, FeatureSpec
from models.SeriesDecomposition import SeriesDecomposition  

class myxLSTM(nn.Module):

    def __init__(
            self, 
            num_classes,
            config
        ): 
        super(myxLSTM, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.patch_size = config.patch_size
        self.multi_token_prediction = config.multi_token_prediction
        self.use_tab_data = config.use_tab_data

        self.series_decomposition = config.series_decomposition
        if config.series_decomposition > 0:
            self.sd = SeriesDecomposition(config.series_decomposition)

        self.activation = get_activation_fn(config.activation_fn)

        self.patch_embedding = PatchEmbedding(patch_size=config.patch_size, num_hiddens=config.embedding_size)

        self.sep_token = nn.Parameter(torch.randn(1, 1, config.embedding_size))

        self.xlstm = get_xlstm(config.embedding_size, dropout=config.dropout, blocks=config.xlstm_config)

        self.fc = nn.Sequential(
            nn.Linear(config.embedding_size, config.embedding_size // 2),
            self.activation,
            nn.Linear(config.embedding_size // 2, num_classes),
        )

        if not self.multi_token_prediction:
            self.reconstruction = HeadModule(config.embedding_size, config.patch_size)
        else:
            self.rec1 = HeadModule(config.embedding_size, config.patch_size, config.dropout)
            self.rec2 = HeadModule(config.embedding_size, config.patch_size, config.dropout)
            self.rec3 = HeadModule(config.embedding_size, config.patch_size, config.dropout)
            self.rec4 = HeadModule(config.embedding_size, config.patch_size, config.dropout)
    
        if self.use_tab_data:
            self.tab_embeddings = TabularEmbeddings([
                FeatureSpec('age', 110, torch.int64),
                FeatureSpec('is_male', 2, torch.bool),
                FeatureSpec('RBBB', 2, torch.bool),
                FeatureSpec('LBBB', 2, torch.bool),
                FeatureSpec('SB', 2, torch.bool), # Sinus Bradycardia
                FeatureSpec('AF', 2, torch.bool), # Atrial Fibrillation
                # FeatureSpec('1dAVb', 2, torch.bool), # first degree AV block, not present in the mit_bih dataset
                # #['age', 'is_male', '1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF', 'normal_ecg'],
            ], num_hiddens=config.embedding_size, dropout=config.dropout)


    def reconstruct(self, x, tab_data):
        if self.series_decomposition > 0:
            res, trend = self.sd(x)
            # add channels 
            x = torch.cat([x, res, trend], dim=-1)

        # mean = mean.squeeze(1)
        # std = std.squeeze(1)
        x = x.permute(0, 2, 1) # put the channels in the middle
        x = self.patch_embedding(x)

        if self.use_tab_data:
            # eventually add the tabular data
            batch_size = x.shape[0]
            tab_emb = self.tab_embeddings(tab_data, batch_size)
            _, num_embeddings, _ = tab_emb.shape
            x = torch.cat([tab_emb, x], dim=1)

        x = self.xlstm(x)

        if self.use_tab_data:
            # remove the tabular data
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
        if self.series_decomposition:
            x = self.sd(x)

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
