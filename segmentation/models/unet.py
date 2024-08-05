from typing import Union, Tuple

import torch
import torch.nn as nn

# Example usage
# --------------
#
# if __name__ == '__main__':
#     model = UNet(depth=4,
#                 base_channel_num=64,
#                 in_channels=1,
#                 out_channels=3)
#     input_tensor = torch.randn(1, 1, 512, 512)  # Example input tensor
#     output_tensor = model(input_tensor)
# 
#     print("Output shape:", output_tensor.shape)
# 

def _crop_and_concat(feature_map1: torch.Tensor, 
                    feature_map2: torch.Tensor) -> torch.Tensor:
    _, _, h1, w1 = feature_map1.size()
    _, _, h2, w2 = feature_map2.size()

    crop_h_start = (h1 - h2) // 2
    crop_h_end = crop_h_start + h2
    crop_w_start = (w1 - w2) // 2
    crop_w_end = crop_w_start + w2

    cropped_feature_map1 = feature_map1[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
    
    return torch.cat((cropped_feature_map1, feature_map2), dim=1)

class UNet(nn.Module):
    def __init__(self,
                 depth : int,
                 base_channel_num : int,
                 in_channels : int,
                 out_channels : int,
                 dropout_rate : float,
                 padding_mode : str) -> None:
        super().__init__()

        self.dropout_rate = dropout_rate

        self.padding_mode = padding_mode

        self.depth = depth
        self.base_channel_num = base_channel_num

        self.in_channels = in_channels
        self.out_channels = out_channels

        # index 0 = top of diagram
        self.down_layers = nn.ModuleList([
            self._make_down_layer(in_channels=self.in_channels,
                                  out_channels=self.base_channel_num,
                                  do_maxpool=False)
        ])

        channel_num = self.base_channel_num

        for i in range(1,self.depth):
            down_layer = self._make_down_layer(in_channels=channel_num)
            self.down_layers.append(down_layer)

            channel_num *= 2

        self.bridge_layer = self._make_bridge_layer(in_channels=channel_num)
        channel_num *= 2

        # index 0 = bottom of diagram
        self.up_layers = nn.ModuleList()
        for i in range(self.depth):
            do_upsample = (i != self.depth - 1)
            up_layer = self._make_up_layer(in_channels=channel_num,
                                           do_upsample=do_upsample)
            self.up_layers.append(up_layer)
            
            channel_num //= 2

        self.conv1x1 = nn.Conv2d(in_channels=channel_num,
                                 out_channels=self.out_channels,
                                 kernel_size=1)

    # A single layer in the left side of the "U"
    def _make_down_layer(self, 
                         in_channels : int,
                         out_channels : Union[int, None] = None,
                         do_maxpool : bool = True) -> nn.Module:
        if out_channels is None:
            out_channels = in_channels * 2

        layers : list
        if do_maxpool:
            layers = [
                nn.MaxPool2d(kernel_size=2,
                         stride=2),
            ]
        else:
            layers = []

        layers += [
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      padding_mode=self.padding_mode),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      padding_mode=self.padding_mode),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_rate),
        ]

        return nn.Sequential(*layers)
    
    def _make_up_layer(self,
                       in_channels : int,
                       do_upsample : bool = True) -> nn.Module:
        out_channels = in_channels // 2

        layers = [
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      padding_mode=self.padding_mode),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      padding_mode=self.padding_mode),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_rate),
        ]

        if do_upsample:
            layers += [
                nn.ConvTranspose2d(in_channels=out_channels, 
                                out_channels=out_channels//2,  #We divide by 2 since there will be a copy+crop with residual channels
                                kernel_size=2, 
                                stride=2)
            ]

        return nn.Sequential(*layers)

    def _make_bridge_layer(self, 
                           in_channels : int) -> nn.Module:
        layers = [
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels*2,
                      kernel_size=3,
                      padding=1,
                      padding_mode=self.padding_mode),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_rate),

            nn.Conv2d(in_channels=in_channels*2,
                      out_channels=in_channels*2,
                      kernel_size=3,
                      padding=1,
                      padding_mode=self.padding_mode),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_rate),

            nn.ConvTranspose2d(in_channels=in_channels*2, 
                                out_channels=in_channels, 
                                kernel_size=2, 
                                stride=2),
        ]

        return nn.Sequential(*layers)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height % (2**self.depth) != 0 or width % (2**self.depth) != 0:
            raise ValueError(f"Invalid image shape for U-Net: {(height, width)}\nH,W should be divisible by 2^depth = {2**self.depth}.")

        residuals = []
        for current_depth in range(self.depth):
            down_layer = self.down_layers[current_depth]

            x = down_layer(x)
            residuals.append(x)
        residuals = residuals[::-1]

        x = self.bridge_layer(x)

        for current_depth in range(self.depth):
            x = _crop_and_concat(feature_map1=residuals[current_depth], 
                                 feature_map2=x)
            up_layer = self.up_layers[current_depth]
            x = up_layer(x)
        
        x = self.conv1x1(x)        

        return x
