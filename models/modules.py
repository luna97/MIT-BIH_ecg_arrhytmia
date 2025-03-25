from torch import nn
import torch 
import numpy as np
from fastonn import SelfONN1d
from models.u_net import DoubleXLSTMDown, DoubleXLSTMUp

class LinearPatchEmbedding(nn.Module):
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
        # print('x shape after deconv', x.shape) [1, 3584, 12]
        return x
    
class ConvPatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12):
        super().__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv1d(num_channels, num_hiddens // 4, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm1d(num_hiddens // 4)
        self.conv2 = nn.Conv1d(num_hiddens // 4, num_hiddens // 2, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(num_hiddens // 2)
        self.conv3 = nn.Conv1d(num_hiddens // 2, num_hiddens, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(num_hiddens)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.activation = nn.ReLU()

        # Calculate the output size after the convolutions and pooling
        out_size = ((patch_size - 7 + 1) // 2 - 5 + 1) // 2 - 3 + 1
        out_size = (out_size // 2) * num_hiddens

        self.linear = nn.Linear(out_size, num_hiddens)

    def forward(self, x):
        # transform [bs, n_channels, n_samples] -> [bs, n_channels, n_patches, patch_size]
        x = x.unfold(2, self.patch_size, self.patch_size).transpose(1, 2)
        batch_size, n_patches, n_channels, _ = x.shape
        x = x.reshape(-1, n_channels, self.patch_size) # [bs * n_patches, n_channels, patch_size]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.flatten(1)
        # print('x shape after conv', x.shape)
        x = self.linear(x)

        x = x.reshape(batch_size, n_patches, -1)

        # print('x shape after unfold', x.shape)
        return x
    

class ONNConvPatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12):
        super().__init__()
        self.patch_size = patch_size
        self.conv1 = SelfONN1d(num_channels, num_hiddens // 4, kernel_size=3, stride=1, q=3)
        self.bn1 = nn.BatchNorm1d(num_hiddens // 4)
        self.conv2 = SelfONN1d(num_hiddens // 4, num_hiddens, kernel_size=3, stride=1, q=3)
        self.bn2 = nn.BatchNorm1d(num_hiddens)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.activation = nn.Tanh()

        # Calculate the output size after the convolutions and pooling
        out_size = (patch_size - 3 + 1) // 2 - 3 + 1
        out_size = (out_size // 2) * num_hiddens

        self.linear = nn.Linear(out_size, num_hiddens)

    def forward(self, x):
        # print('x shape', x.shape)
        # transform [bs, n_channels, n_samples] -> [bs, n_channels, n_patches, patch_size]
        x = x.unfold(2, self.patch_size, self.patch_size).transpose(1, 2)
        batch_size, n_patches, n_channels, _ = x.shape
        x = x.view(-1, n_channels, self.patch_size) # [bs * n_patches, n_channels, patch_size]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.flatten(1)
        # print('x shape after conv', x.shape)
        x = self.linear(x)

        x = x.view(batch_size, n_patches, -1)

        # print('x shape after unfold', x.shape)
        return x
    
class UNetPatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12, dropout=0.1):
        super().__init__()
        self.linear_patch_emb = LinearPatchEmbedding(patch_size, num_hiddens, num_channels)

        self.down1 = DoubleXLSTMDown(num_hiddens, final_size=num_hiddens // 2, dropout=dropout)
        self.down2 = DoubleXLSTMDown(num_hiddens // 2, final_size=num_hiddens // 4, dropout=dropout)
        # self.down3 = DoubleXLSTMDown(num_hiddens // 4, final_size=num_hiddens // 8, dropout=dropout)
        self.patch_size = patch_size

    def forward(self, x):
        x2 = self.linear_patch_emb(x)
        x1 = self.down1(x2)
        x = self.down2(x1)
        # x3 = self.down3(x2)
        return x, x1, x2, None
  
    
class UNetEmbedPatching(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12):
        super().__init__()
        self.up1 = DoubleXLSTMUp(num_hiddens // 2, initial_size=num_hiddens // 4, dropout=0.1)
        self.up2 = DoubleXLSTMUp(num_hiddens, initial_size=num_hiddens // 2, dropout=0.1)
        # self.up3 = DoubleXLSTMUp(num_hiddens, initial_size=num_hiddens // 2, dropout=0.1)
        self.patch_size = patch_size
        self.depatch = EmbedPatching(patch_size, num_hiddens, num_channels)

    def forward(self, x, x1, x2, _):
        # x after middle block
        #print('x shape', x.shape)
        #print('x1 shape', x1.shape)
        #print('x2 shape', x2.shape)
        x = self.up1(x, x1)
        x = self.up2(x, x2)
        # x = self.up3(x, x1)
        x = self.depatch(x)
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
        # print('tab_data', tab_data)
        device = next(self.parameters()).device
        # zero_tensor = torch.zeros(batch_size, self.num_hiddens, device=device)
        # print('type of tab_data', type(tab_data))
        for feat in self.feature_specs:
            values = tab_data.get(feat.name)
            if values is not None:
                values = torch.as_tensor(values.values.astype(float), dtype=feat.dtype).to(device)
                if not (values == 0).all():
                    values = values.clamp_max(feat.num_categories * feat.category_size - 1) // feat.category_size  # Ensure values are within range
                    values = values.to(torch.int32)
                    embeddings.append(self.embeddings[feat.name](values))

        embeddings = [emb for emb in embeddings if emb is not None]

        if embeddings is None or len(embeddings) == 0:
            return None
        
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
            nn.ReLU(activation_fn),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_size),
        )
        
    def forward(self, x):
        return self.head(x)
    
class mLSTMWrapper(nn.Module):
    def __init__(self, xlstm, dropout=0.2):
        super(mLSTMWrapper, self).__init__() 
        self.model = xlstm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        len_seq = x.shape[1]
        pad_len = max(16 - len_seq, 2**int(np.ceil(np.log2(len_seq))) - len_seq)
        x = torch.cat([torch.zeros(x.shape[0], pad_len, x.shape[2]).to(x.device), x], dim=1)
        x, _ = self.model_forward_wrap(x)
        return x[:, pad_len:, :]
    
    def step(self, x, state):
        len_seq = x.shape[1]
        pad_len = max(16 - len_seq, 2**int(np.ceil(np.log2(len_seq))) - len_seq)
        x = torch.cat([torch.zeros(x.shape[0], pad_len, x.shape[2]).to(x.device), x], dim=1)
        x, state = self.model_forward_wrap(x, state)
        return x[:, pad_len:, :], state
    
    def model_forward_wrap(self, x, state = None):
        if state is None:
            state = {i: None for i in range(len(self.model.blocks))}

        for i, block in enumerate(self.model.blocks):
            block_state = state[i]
            x = self.dropout(x)
            x, block_state_new = block(x, block_state)

            if block_state is None:
                state[i] = block_state_new
            else:
                # layer state is a tuple of three tensors: c, n, m
                # we update the state in place in order to avoid creating new tensors
                for state_idx in range(len(block_state)):
                    state[i][state_idx].copy_(block_state_new[state_idx])

        x = self.model.out_norm(x)

        return x, state