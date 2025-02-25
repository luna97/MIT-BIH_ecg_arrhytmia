from torch import nn
import torch 
import numpy as np
from torch.nn import functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256):
        super().__init__()

        self.conv = nn.LazyConv1d(num_hiddens, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # print('X shape', X.shape)
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(x).flatten(2).transpose(1, 2)


class FeatureSpec(object):
    def __init__(self, name, num_categories, dtype):
        self.name = name
        self.num_categories = num_categories
        self.dtype = dtype
        torch.fft
 

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
        zero_tensor = torch.zeros(batch_size, self.num_hiddens, device=device)
        # print('type of tab_data', type(tab_data))
        for feat in self.feature_specs:
            values = tab_data.get(feat.name)
            # if the data do not have the feature, we add a tensor of zeros
            if values is None:
                embeddings.append(zero_tensor)
            else:
                # print(values)
                values = torch.as_tensor(values.values, dtype=feat.dtype).to(device)
                values = values.clamp_max(feat.num_categories - 1)  # Ensure values are within range
                embeddings.append(self.embeddings[feat.name](values))
        
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