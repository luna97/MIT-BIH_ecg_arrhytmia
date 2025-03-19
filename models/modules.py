from torch import nn
import torch 
import numpy as np
from torch.nn import functional as F
from models.utils import get_activation_fn
from joblib import Parallel, delayed
from dataset.generic_utils import get_max_n_jobs

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12):
        super().__init__()
        self.conv = nn.Conv1d(num_channels, num_hiddens, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x = self.conv(x).flatten(2).transpose(1, 2)
        return x
    
    
class EmbedPatching(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12, activation_fn='relu', use_pre_head=False):
        super().__init__()
        self.use_pre_head = use_pre_head  
        if use_pre_head: self.pre_head = HeadModule(num_hiddens, num_hiddens // 2, num_hiddens, activation_fn=activation_fn)
        self.deconv = nn.ConvTranspose1d(num_hiddens, num_channels, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        if self.use_pre_head: x = self.pre_head(x)
        x = x.transpose(1, 2)
        x = self.deconv(x).transpose(1, 2)
        return x
    

class FeatureSpec(object):
    def __init__(self, name, num_categories, dtype, category_size=1):
        self.name = name
        self.num_categories = num_categories
        self.dtype = dtype
        self.category_size = category_size
 

class TabularEmbeddings(nn.Module):
    def __init__(self, feature_specs, num_hiddens=256, dropout=0.1):
        """
        feature_specs: dict of {column_name: (num_categories, dtype)}
        Example: {"is_male": (2, torch.int64), "age": (110, torch.int64)}
        """
        super().__init__()
        print('creating embeddings')
        self.embeddings = nn.ModuleDict({
            feat.name: nn.Embedding(feat.num_categories, num_hiddens)
            for feat in feature_specs
        })
        self.feature_specs = feature_specs
        self.num_hiddens = num_hiddens
        self.dropout = FeatureDropout(dropout)

    def forward(self, tab_data, batch_size):
        embeddings = []
        device = next(self.parameters()).device
        # zero_tensor = torch.zeros(batch_size, self.num_hiddens, device=device)
        # print('type of tab_data', type(tab_data))
        def process_feature(feat):
            values = tab_data.get(feat.name)
            if values is not None:
                values = torch.as_tensor(values.values.astype(float), dtype=feat.dtype).to(device)
                if not (values == 0).all():
                    values = values.clamp_max(feat.num_categories * feat.category_size - 1) // feat.category_size  # Ensure values are within range
                    values = values.to(torch.int32)
                    return self.embeddings[feat.name](values)
                return None

        embeddings = Parallel(n_jobs=get_max_n_jobs())(delayed(process_feature)(feat) for feat in self.feature_specs)
        embeddings = [emb for emb in embeddings if emb is not None]

        if embeddings == []:
            tortn = torch.zeros(batch_size, 1, self.num_hiddens, device=device)
        
        tortn = torch.stack(embeddings, dim=1) # (batch_size, num_features, num_hiddens)
        tortn = self.dropout(tortn)
        return tortn
    

class FeatureDropout(nn.Module):
    """
    Apply dropout to the embeddings of the tabular data
    Zero out entire embeddings with a certain probability
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:  # Apply only during training
            return self.random_feature_dropout(x, self.p)
        else:
            return x  # Do nothing during evaluation (inference)

    def random_feature_dropout(self, embeddings, p):
        batch_size, num_features, num_hiddens = embeddings.shape
        mask = torch.rand((batch_size, num_features), device=embeddings.device) > p # generates the mask on the same device as embeddings
        mask = mask.unsqueeze(-1).expand(-1, -1, num_hiddens)
        return embeddings * mask
    

class HeadModule(nn.Module):
    
    def __init__(self, inp_size, hidden_size, out_size, activation_fn='relu', dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            get_activation_fn(activation_fn),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_size),
        )
        
    def forward(self, x):
        return self.head(x)