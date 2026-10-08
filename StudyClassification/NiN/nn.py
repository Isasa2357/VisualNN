# Network in Network (NiN)の実装
# 入力はCIFAR(32x32)を想定

import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from comm.nn import param_init

class NiN(nn.Module):
    def __init__(self, class_num: int, dropout: float=0.5):
        super().__init__()

        self._mlpconv1 = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, padding=2, stride=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 
            nn.Dropout(dropout)
        )

        self._mlpconv2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 
            nn.Dropout(dropout)
        )

        self._mlpconv3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, class_num, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self._classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten()
        )

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mlpconv1(x)
        x = self._mlpconv2(x)
        x = self._mlpconv3(x)
        x = self._classifier(x)
        return x