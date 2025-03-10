import torch
from torch import nn

from models.utils import get_activation_fn, get_xlstm
from models.PatchEmbedding import TabularEmbeddings, PatchEmbedding, FeatureSpec, EmbeddingToPatch
from models.SeriesDecomposition import SeriesDecomposition 
import numpy as np

class myxLSTM(nn.Module):

    def __init__(
            self, 
            num_classes,
            num_channels,
            config
        ): 
        super(myxLSTM, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.patch_size = config.patch_size
        self.multi_token_prediction = config.multi_token_prediction
        self.use_tab_data = config.use_tab_data
        self.weight_tying = config.weight_tying


        self.activation = get_activation_fn(config.activation_fn)

        self.patch_embedding = PatchEmbedding(patch_size=config.patch_size, num_hiddens=config.embedding_size)

        self.sep_token = nn.Parameter(torch.randn(1, 1, config.embedding_size))

        self.xlstm = get_xlstm(config.embedding_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads)

        self.random_drop_leads = RandomDropLeads(config.random_drop_leads)

        self.fc = nn.Sequential(
            nn.Linear(config.embedding_size, config.embedding_size // 2),
            self.activation,
            nn.Linear(config.embedding_size // 2, num_classes),
        )

        if not self.weight_tying:
            if not self.multi_token_prediction:
                self.reconstruction = HeadModule(config.embedding_size, config.patch_size, num_channels, config.dropout)
            else:
                self.rec1 = HeadModule(config.embedding_size, config.patch_size, num_channels, config.dropout)
                self.rec2 = HeadModule(config.embedding_size, config.patch_size, num_channels, config.dropout)
                self.rec3 = HeadModule(config.embedding_size, config.patch_size, num_channels, config.dropout)
                self.rec4 = HeadModule(config.embedding_size, config.patch_size, num_channels, config.dropout)
    
        if self.use_tab_data:
            self.tab_embeddings = TabularEmbeddings([
                FeatureSpec('age', 110, torch.int64),
                FeatureSpec('is_male', 2, torch.bool),
                #FeatureSpec('RBBB', 2, torch.bool),
                #FeatureSpec('LBBB', 2, torch.bool),
                #FeatureSpec('SB', 2, torch.bool), # Sinus Bradycardia
                #FeatureSpec('AF', 2, torch.bool), # Atrial Fibrillation
                # FeatureSpec('1dAVb', 2, torch.bool), # first degree AV block, not present in the mit_bih dataset
                # #['age', 'is_male', '1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF', 'normal_ecg'],
            ], num_hiddens=config.embedding_size, dropout=config.dropout)

    def embed_data(self, x, tab_data):
        x = self.random_drop_leads(x)

        x = x.permute(0, 2, 1) # put the channels in the middle
        x = self.patch_embedding(x)

        if self.use_tab_data and tab_data is not None:
            # eventually add the tabular data
            batch_size = x.shape[0]
            tab_emb = self.tab_embeddings(tab_data, batch_size)
            _, num_embeddings, _ = tab_emb.shape
            x = torch.cat([tab_emb, x], dim=1)
            return x, num_embeddings
        else:
            return x, 0

    def reconstruct(self, x, tab_data):
        x, tab_embeddings = self.embed_data(x, tab_data)

        x = self.xlstm(x)

        if self.use_tab_data:
            # remove the tabular data
            x = x[:, tab_embeddings:, :]

        if self.weight_tying:
            x = self.patch_embedding.get_patch(x)
            return x
        
        if self.multi_token_prediction:
            x1 = self.rec1(x)
            x2 = self.rec2(x)
            x3 = self.rec3(x)
            x4 = self.rec4(x)
            return x1, x2, x3, x4
        
        x = self.reconstruction(x)
        return x
    
    def generate(self, x, tab_data, length=10):
        x, _ = self.embed_data(x, tab_data)

        state = None
        for i in range(x.shape[1]):
            new_x, state = self.xlstm.step(x[:, i].unsqueeze(1), state=state)

        if self.weight_tying:
            r = self.patch_embedding.get_patch(new_x)
        elif self.multi_token_prediction:
            r = self.rec1(new_x)
        else:
            r = self.reconstruction(new_x)

        reconstructed = [r]

        for i in range(length - 1):
            x, _ = self.embed_data(r, None)
            x, state = self.xlstm.step(x, state=state)
            if self.weight_tying:
                r = self.patch_embedding.get_patch(x)
            elif self.multi_token_prediction:
                r = self.rec1(x)
            else:
                r = self.patch_embedding.get_patch(x)
            reconstructed.append(r)
        
        return torch.cat(reconstructed, dim=1)


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
    

class RandomDropLeads(nn.Module):
    def __init__(self, probability=0.5):
        super(RandomDropLeads, self).__init__()
        self.probability = probability

    def forward(self, signal):
        if self.train:
            leads_to_remove = np.random.random(signal.shape[-1]) < self.probability
            leads_to_remove[1] = False  # never remove lead II
            signal[..., leads_to_remove] = 0
        return signal
    
class HeadModule(nn.Module):
    def __init__(self, in_features, out_features, num_channels=1, dropout=0.3):
        super(HeadModule, self).__init__()
        self.num_channels = num_channels 
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, out_features * num_channels),
        )

    def forward(self, x):
        x = self.fc(x) # [batch_size, patch_size, out_dim * num_channels]
        x = x.view(x.shape[0], -1, self.num_channels) # [batch_size, seq_len, num_channels]
        return x
