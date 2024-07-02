import torch
import torch.nn as nn

#Works on input of shape (batch, channels, [Image Shape])
class TileChannels(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n

    def forward(self, x):
        shape = x.shape
        tiled_x = x.repeat(1, self.n, *([1] * (len(shape) - 2)))
        
        return tiled_x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        
        if stride not in [1,2]:
            raise ValueError(f"Stride for a ResNet Block must be 1 or 2, not {stride}")
        self.downsample = (stride == 2)
        
        if self.downsample:
            self.downsample_pool = nn.MaxPool2d(kernel_size=2)
            self.downsample_tile_channels = TileChannels(n=2)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=self.out_channels,
                               out_channels=self.out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu2 = nn.ReLU() 
        
        self.relu3 = nn.ReLU() 

    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        #Fix skip connection to match shape of transformed x
        if self.downsample:
            #Fix image size
            identity = self.downsample_pool(identity)
            #Fix number of channels
            identity = self.downsample_tile_channels(identity)
            
        x = x + identity
        
        x = self.relu3(x)
        
        return x
