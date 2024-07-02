import torch.nn as nn

from resnet_block import ResNetBlock

# Can only process input image size multiple of (32, 32)
class ResNet(nn.Module):
    def __init__(self, layers, in_channels=1, num_classes=2) -> None:
        if len(layers) != 4:
            raise ValueError(f"ResNet only supports 4 layers, not the provided layer list {layers} of length {len(layers)}")
            
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               padding_mode='circular')
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        
        self.layer1 = self._make_layer(in_channels=64, 
                                       out_channels=64, 
                                       num_blocks=layers[0],
                                       downsample=False)
        
        self.layer2 = self._make_layer(in_channels=64, 
                                       out_channels=128, 
                                       num_blocks=layers[1],
                                       downsample=True)
        
        self.layer3 = self._make_layer(in_channels=128, 
                                       out_channels=256, 
                                       num_blocks=layers[2],
                                       downsample=True)
        
        self.layer4 = self._make_layer(in_channels=256, 
                                       out_channels=512, 
                                       num_blocks=layers[3],
                                       downsample=True)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, downsample) -> nn.Sequential:
        layers = []
        
        for i in range(num_blocks):
            stride = 2 if (i == 0 and downsample) else 1
            local_in_channels = in_channels if (i==0) else out_channels
            local_out_channels = out_channels
            
            layer = ResNetBlock(in_channels=local_in_channels, out_channels=local_out_channels, stride=stride)
            
            layers.append(layer)
        
        return nn.Sequential(*layers)

    def forward(self, x):
        #shape of x = (Batch, Channels, Height, Width)
        if len(x.shape) != 4:
            raise ValueError(f"ResNet only processes input of shape (batch, channels, height, width), not {tuple(x.shape)}")
        B, C, H, W = x.shape
        if W % 32 != 0 or H % 32 != 0:
            raise ValueError(f"ResNet can only process images whose dimensions are divisible by 32, not {W} x {H}. \n Full Input shape: {tuple(x.shape)}")
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = self.lin(x)
        
        return x
        