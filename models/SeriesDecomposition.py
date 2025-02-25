from torch import nn
import torch

class MovingAvg(nn.Module):
    def __init__(self, moving_avg):
        super(MovingAvg, self).__init__()
        self.kernel_size = moving_avg
        self.avg = nn.AvgPool1d(kernel_size=moving_avg, stride=1, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1, 1)
        # end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x        
    

class SeriesDecomposition(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean