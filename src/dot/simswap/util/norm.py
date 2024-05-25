#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn


class SpecificNorm(nn.Module):
    def __init__(self, epsilon=1e-8, use_gpu=True):
        """
        @notice: avoid in-place ops.
        https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(SpecificNorm, self).__init__()
        self.mean = np.array([0.485, 0.456, 0.406])
        if use_gpu:
            self.mean = (
                torch.from_numpy(self.mean)
                .float()
                .to("mps" if torch.backends.mps.is_available() else "cuda")
            )
        else:
            self.mean = torch.from_numpy(self.mean).float().cpu()

        self.mean = self.mean.view([1, 3, 1, 1])

        self.std = np.array([0.229, 0.224, 0.225])
        if use_gpu:
            self.std = (
                torch.from_numpy(self.std)
                .float()
                .to("mps" if torch.backends.mps.is_available() else "cuda")
            )
        else:
            self.std = torch.from_numpy(self.std).float().cpu()

        self.std = self.std.view([1, 3, 1, 1])

    def forward(self, x, use_gpu=True):
        mean = self.mean.expand([1, 3, x.shape[2], x.shape[3]])
        std = self.std.expand([1, 3, x.shape[2], x.shape[3]])

        if use_gpu:
            x = (x - mean) / std
        else:
            x = (x - mean.detach().to("cpu")) / std.detach().to("cpu")

        return x
