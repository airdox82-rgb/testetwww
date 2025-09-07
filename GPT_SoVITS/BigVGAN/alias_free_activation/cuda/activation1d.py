# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import torch
import torch.nn as nn
from alias_free_activation.torch.resample import UpSample1d, DownSample1d

class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        fused: bool = True,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

        # Force fused to False if CUDA is not available
        self.fused = fused and torch.cuda.is_available()

    def forward(self, x):
        if not self.fused:
            x = self.upsample(x)
            x = self.act(x)
            x = self.downsample(x)
            return x
        else:
            raise RuntimeError("Fused CUDA kernel is not supported in CPU-only mode.")