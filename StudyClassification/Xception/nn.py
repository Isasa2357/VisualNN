# Xceptionの実装

import torch
from torch import nn
import torch.nn.functional as F

from comm.nn import param_init

class SeparableConv(nn.Module):
    '''
    XceptionのSeparableConv層
    '''

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int
                 ):
        super().__init__()

        self._depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            3, 1, 1, groups=in_channels, 
            bias=False
        )
        self._pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            1, 1, 0, 
            bias=False
        )
        self._bn = nn.BatchNorm2d(out_channels)

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._depthwise(x)
        x = self._pointwise(x)
        x = self._bn(x)
        return x

class XceptionEntryFlow(nn.Module):

    def __init__(self):
        super().__init__()
        self._initial_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self._block1 = nn.Sequential(
            SeparableConv(64, 128),
            nn.ReLU(inplace=True), 
            SeparableConv(128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self._block1_residual = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, 0, bias=False),
            nn.BatchNorm2d(128)
        )

        self._block2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv(128, 256),
            nn.ReLU(inplace=True),
            SeparableConv(256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self._block2_residual = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, 0, bias=False),
            nn.BatchNorm2d(256)
        )

        self._block3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv(256, 728),
            nn.ReLU(inplace=True),
            SeparableConv(728, 728),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self._block3_residual = nn.Sequential(
            nn.Conv2d(256, 728, 1, 2, 0, bias=False),
            nn.BatchNorm2d(728)
        )

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._initial_block(x)

        # reluのinplace演算があるため，主経路と残差経路に分けてxを保持
        residual_x = x.clone()
        residual = self._block1_residual(residual_x)
        x = self._block1(x)
        x += residual

        residual_x = x.clone()
        residual = self._block2_residual(residual_x)
        x = self._block2(x)
        x += residual

        residual_x = x.clone()
        residual = self._block3_residual(residual_x)
        x = self._block3(x)
        x += residual

        return x

class XceptionMiddleFlow(nn.Module):

    def __init__(self):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv(728, 728),
            nn.ReLU(inplace=True),
            SeparableConv(728, 728),
            nn.ReLU(inplace=True),
            SeparableConv(728, 728),
        )

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.clone()
        return self._block(x) + residual

class XceptionExitFlow(nn.Module):

    def __init__(self):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv(728, 728),
            nn.ReLU(inplace=True),
            SeparableConv(728, 1024),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self._block_residual = nn.Sequential(
            nn.Conv2d(728, 1024, 1, 2, 0, bias=False),
            nn.BatchNorm2d(1024)
        )

        self._final_block = nn.Sequential(
            SeparableConv(1024, 1536),
            nn.ReLU(inplace=True),
            SeparableConv(1536, 2048), 
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((1, 1))
        )

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_x = x.clone()
        residual = self._block_residual(residual_x)
        x = self._block(x)
        x += residual

        x = self._final_block(x)

        return x

class Xception(nn.Module):

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()

        self._entry_flow = XceptionEntryFlow()
        self._middle_flows = nn.Sequential(
            XceptionMiddleFlow(),
            XceptionMiddleFlow(),
            XceptionMiddleFlow(),
            XceptionMiddleFlow(),
            XceptionMiddleFlow(),
            XceptionMiddleFlow(),
            XceptionMiddleFlow(), 
            XceptionMiddleFlow()
        )
        self._exit_flow = XceptionExitFlow()

        self._classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._entry_flow(x)
        x = self._middle_flows(x)
        x = self._exit_flow(x)
        x = self._classifier(x)
        return x
