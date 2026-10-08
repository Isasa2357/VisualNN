
import torch
from torch import nn

class DepthwiseSeparableConv(nn.Module):

    '''
    空間方向の畳み込みとチャネル方向の畳み込みを分離した畳み込み層
    '''

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):

        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        self._depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2 if downsample else 1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(inplace=True)
        )

        self._pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 4次元テンソルかつチャネル数が入力チャンネル数と一致することを確認
        if x.dim() != 4 or x.size(1) != self._in_channels:
            raise ValueError(f"Input tensor must have shape (batch_size, {self._in_channels}, height, width)")
        
        x = self._depthwise(x)
        x = self._pointwise(x)
        return x

class MobileNetV1(nn.Module):

    def __init__(self, num_classes: int):

        super().__init__()

        self._initial_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True)
        )

        self._depthwise_layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128, downsample=True), 
            DepthwiseSeparableConv(128, 128), 
            DepthwiseSeparableConv(128, 256, downsample=True), 
            DepthwiseSeparableConv(256, 256), 
            DepthwiseSeparableConv(256, 512, downsample=True), 
            DepthwiseSeparableConv(512, 512), 
            DepthwiseSeparableConv(512, 512), 
            DepthwiseSeparableConv(512, 512), 
            DepthwiseSeparableConv(512, 512), 
            DepthwiseSeparableConv(512, 512), 
            DepthwiseSeparableConv(512, 1024, downsample=True), 
            DepthwiseSeparableConv(1024, 1024)
        )

        self._classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=7),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # バッチ次元付き2.5次元テンソルのみ受け入れ
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError("Input tensor must have shape (batch_size, 3, height, width)")

        x = self._initial_layers(x)
        x = self._depthwise_layers(x)
        x = self._classifier(x)
        return x