import torch
from torch import nn

from models.utils import get_activation_fn, get_xlstm, get_large_xstm
from models.modules import TabularEmbeddings, PatchEmbedding, FeatureSpec, EmbedPatching, HeadModule
from models.SeriesDecomposition import SeriesDecomposition 
from augmentations import RandomDropLeads, FTSurrogate, Jitter
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
        self.use_tab_data = config.use_tab_data
        self.weight_tying = config.weight_tying


        self.activation = get_activation_fn(config.activation_fn)

        self.patch_embedding = PatchEmbedding(patch_size=config.patch_size, num_hiddens=config.embedding_size)
        self.sep_token = nn.Parameter(torch.randn(1, 1, config.embedding_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embedding_size))

        self.xlstm = get_xlstm(config.embedding_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads)

        self.random_drop_leads = RandomDropLeads(config.random_drop_leads)
        self.random_surrogate = FTSurrogate(0.05, prob=config.random_surrogate_prob)
        self.random_jitter = Jitter(sigma=0.1, prob=config.random_jitter_prob)

        self.fc = HeadModule(
            inp_size=config.embedding_size,
            hidden_size=config.embedding_size // 2, 
            out_size=num_classes, 
            dropout=config.dropout, 
            activation_fn=config.activation_fn
        )

        self.use_mean_var_head = 'mean_var' in config.loss_type

        if self.use_mean_var_head:
            self.variance_head = HeadModule(
                inp_size=config.embedding_size, 
                hidden_size=config.embedding_size // 2,
                out_size=1, 
                dropout=config.dropout,
                activation_fn=config.activation_fn
            )
            self.mean_head = HeadModule(
                inp_size=config.embedding_size, 
                hidden_size=config.embedding_size // 2, 
                out_size=1, 
                dropout=config.dropout,
                activation_fn=config.activation_fn
            )

        self.reconstruction = EmbedPatching(
            patch_size=config.patch_size, 
            num_hiddens=config.embedding_size, 
            num_channels=len(config.leads), 
            activation_fn=config.activation_fn, 
            use_pre_head=True
        )

        if self.weight_tying: self.reconstruction.deconv.weight = self.patch_embedding.conv.weight

    
        if self.use_tab_data:
            self.tab_embeddings = TabularEmbeddings([
                FeatureSpec('age', 11, torch.int64, category_size=10),
                FeatureSpec('is_male', 2, torch.bool),                
                FeatureSpec('RBBB', 2, torch.bool),
                FeatureSpec('LBBB', 2, torch.bool),
                FeatureSpec('SB', 2, torch.bool), # Sinus Bradycardia
                FeatureSpec('ST', 2, torch.bool), # Sinus Tachycardia
                FeatureSpec('AF', 2, torch.bool), # Atrial Fibrillation
                FeatureSpec('1dAVb', 2, torch.bool), # first degree AV block, not present in the mit_bih dataset
                FeatureSpec('Hypertension', 2, torch.bool), # 
                FeatureSpec('Ischaemic disease', 2, torch.bool), # 
                FeatureSpec('Pulmonary Heart', 2, torch.bool), #
                FeatureSpec('Cerebrovascular diseases', 2, torch.bool), #
                FeatureSpec('Arteries diseases', 2, torch.bool), # 
                FeatureSpec('Veins diseases', 2, torch.bool), #
                FeatureSpec('Heart Failure', 2, torch.bool), #
                FeatureSpec('Cardiomiopathy', 2, torch.bool), #
                FeatureSpec('Rheumatic disease', 2, torch.bool), #
            ], num_hiddens=config.embedding_size, dropout=config.dropout)

    def embed_data(self, x, tab_data, augment=True):
        if augment:
            x = self.random_drop_leads(x)
            x = self.random_surrogate(x)
            x = self.random_jitter(x)

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
        if isinstance(x, tuple):
            x = x[0]

        if self.use_tab_data:
            # remove the tabular data
            x = x[:, tab_embeddings:, :]
        
        r = self.reconstruction(x)

        if self.use_mean_var_head:
            mean = self.mean_head(x[:, -1, :])
            variance = self.variance_head(x[:, -1, :])
            return r, mean, variance
    
        return r
    
    def generate(self, x, tab_data, length=10):
        # i do not need to drop the leads here
        x, _ = self.embed_data(x, tab_data, augment=False)

        state = None
        for i in range(x.shape[1]):
            new_x, state = self.xlstm.step(x[:, i].unsqueeze(1), state=state)

        r = self.reconstruction(new_x)

        reconstructed = [r]

        for i in range(length - 1):
            x, _ = self.embed_data(r, None, augment=False)
            x, state = self.xlstm.step(x, state=state)
            r = self.reconstruction(x)

            reconstructed.append(r)
        
        return torch.cat(reconstructed, dim=1)


    def forward(self, ctx, x, tab_data):
        ctx, _ = self.embed_data(ctx, tab_data)
        x, _ = self.embed_data(x, None)

        # add the separation token between the context and the input
        sep_token = self.sep_token.repeat(x.shape[0], 1, 1)
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([ctx, sep_token, x, cls_token], dim=1)
        # get the last hidden state and apply the head
        x = self.xlstm(x) # [batch_size, embedding_dim]
        if isinstance(x, tuple):
            x = x[0]
        
        cls_token = x[:, -1, :]
        x = self.fc(cls_token)
        return x, cls_token

    def trainable_parameters(self):
        return self.parameters()
    
    def head_parameters(self):
        return self.fc.parameters() 
    
