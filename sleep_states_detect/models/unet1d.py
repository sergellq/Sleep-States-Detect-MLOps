import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

# Define the layers as before, such as ConvBNReLU, SEBlock, ResBlock, etc.


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        if stride == 1:
            padding = "same"
        else:
            padding = (kernel_size - stride) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class SEBlock(nn.Module):
    def __init__(self, n_channels, se_ratio):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Conv1d(n_channels, n_channels // se_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_channels // se_ratio, n_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, se_ratio):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBNReLU(n_channels, n_channels, kernel_size, stride=1),
            ConvBNReLU(n_channels, n_channels, kernel_size, stride=1),
            SEBlock(n_channels, se_ratio),
        )

    def forward(self, x):
        x_re = self.layers(x)
        return x + x_re


class UNet1d(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.down_kernel_size = cfg["down_kernel_size"]
        self.down_stride = cfg["down_stride"]

        # initial_layers
        self.initial_layers = ConvBNReLU(
            cfg["input_channels"],
            cfg["initial_channels"],
            cfg["initial_kernel_size"],
            stride=1,
            groups=cfg["input_channels"],
        )

        # down_layers
        down_layers = []
        for i in range(len(cfg["down_channels"])):
            in_channels = (
                cfg["down_channels"][i - 1] + cfg["input_channels"]
                if i > 0
                else cfg["initial_channels"]
            )
            out_channels = cfg["down_channels"][i]
            kernel_size = cfg["down_kernel_size"][i]
            stride = cfg["down_stride"][i]
            block = [ConvBNReLU(in_channels, out_channels, kernel_size, stride)]
            for _ in range(cfg["res_depth"]):
                block.append(
                    ResBlock(out_channels, cfg["res_kernel_size"], cfg["se_ratio"])
                )
            down_layers.append(nn.Sequential(*block))
        self.down_layers = nn.ModuleList(down_layers)

        # up_layers
        up_layers = []
        for i in range(len(cfg["down_channels"]) - 1, 0, -1):
            in_channels = cfg["down_channels"][i] + cfg["down_channels"][i - 1]
            up_layers.append(
                ConvBNReLU(
                    in_channels,
                    cfg["down_channels"][i - 1],
                    cfg["down_kernel_size"][i],
                    stride=1,
                )
            )
        self.up_layers = nn.ModuleList(up_layers)

        # out_layers
        self.out_layers = nn.Conv1d(
            cfg["down_channels"][1], 1, cfg["out_kernel_size"], padding="same"
        )

    def forward(self, x):
        """
        Forward pass of the U-Net1D model. Applies initial convolution, downsampling,
        residual blocks, upsampling, and final output layer.
        """
        outs = []  # To store outputs from the downsampling layers
        x_avg = x  # For average pooling in skip connections
        x = self.initial_layers(x)

        # Downsampling
        for i in range(len(self.down_layers)):
            x_out = self.down_layers[i](x)
            if i == len(self.down_layers) - 1:
                x = x_out
            else:
                outs.append(x_out)
                kernel_size = self.down_kernel_size[i]
                stride = self.down_stride[i]
                padding = (kernel_size - stride) // 2
                x_avg = F.avg_pool1d(x_avg, kernel_size, stride, padding)
                x = torch.cat([x_out, x_avg], dim=1)

        # Upsampling
        for i in range(len(self.up_layers)):
            scale_factor = self.down_stride[-i - 1]
            x = F.interpolate(x, scale_factor=scale_factor, mode="linear")
            x = torch.cat([x, outs[-i - 1]], dim=1)
            x = self.up_layers[i](x)

        # Final output layer
        x_out = self.out_layers(x)
        x_out = x_out[:, 0, 180:-180]  # Cropping the output to remove extra padding
        return x_out
