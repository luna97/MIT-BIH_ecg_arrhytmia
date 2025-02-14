from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256):
        super().__init__()

        self.conv = nn.LazyConv1d(num_hiddens, kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        print('X shape', X.shape)
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
