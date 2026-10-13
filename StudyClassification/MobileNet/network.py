
import torch
from torch import nn

class DepthwiseSeparableConv(nn.Module):

    '''
    空間方向の畳み込みとチャネル方向の畳み込みを分離した畳み込み層
    '''

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False, activate_fn: type[nn.Module] = nn.ReLU):

        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        self._depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2 if downsample else 1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), 
            activate_fn(inplace=True)
        )

        self._pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            activate_fn(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
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

class MobileNetV2Block(nn.Module):
    '''
    MobileNetV2の基本ブロック（拡張->深さ方向畳み込み->ポイントワイズ畳み込み）を実装するクラス。
    residual_connectionがTrueの場合、入力と出力のチャネル数が一致している必要がある。
    downsampleがTrueの場合、深さ方向畳み込みで空間解像度を半分にする。
    '''

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, downsample: bool=False, residual_connection: bool=False):
        super().__init__()

        if (residual_connection and in_channels != out_channels) or (residual_connection and downsample):
            raise ValueError("Residual connection requires in_channels to be equal to out_channels and no downsampling")
        self._residual_connection = residual_connection

        self._expand_layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True)
        )

        self._depthwise_layer = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2 if downsample else 1, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True)
        )

        self._pointwise_layer = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self._expand_layer(x)
        x = self._depthwise_layer(x)
        x = self._pointwise_layer(x)
        if self._residual_connection:
            x += identity
        return x

class MobileNetV2(nn.Module):

    def __init__(self, num_classes: int, expand_rate: float=6.0, dropout_rate: float=0.2):
        super().__init__()

        self._feature_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            #
            MobileNetV2Block(16, int(16 * expand_rate), 24, downsample=True), 
            MobileNetV2Block(24, int(24 * expand_rate), 24, residual_connection=True),
            MobileNetV2Block(24, int(24 * expand_rate), 32, downsample=True),
            MobileNetV2Block(32, int(32 * expand_rate), 32, residual_connection=True),
            MobileNetV2Block(32, int(32 * expand_rate), 32, residual_connection=True), 
            MobileNetV2Block(32, int(32 * expand_rate), 64, downsample=True), 
            MobileNetV2Block(64, int(64 * expand_rate), 64, residual_connection=True), 
            MobileNetV2Block(64, int(64 * expand_rate), 64, residual_connection=True), 
            MobileNetV2Block(64, int(64 * expand_rate), 64, residual_connection=True), 
            MobileNetV2Block(64, int(64 * expand_rate), 96), 
            MobileNetV2Block(96, int(96 * expand_rate), 96, residual_connection=True), 
            MobileNetV2Block(96, int(96 * expand_rate), 96, residual_connection=True),
            MobileNetV2Block(96, int(96 * expand_rate), 160, downsample=True),
            MobileNetV2Block(160, int(160 * expand_rate), 160, residual_connection=True),
            MobileNetV2Block(160, int(160 * expand_rate), 160, residual_connection=True),
            MobileNetV2Block(160, int(160 * expand_rate), 320), 
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280), 
            nn.ReLU6(inplace=True)
        )

        self._classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._feature_layers(x)
        x = self._classifier(x)
        return x
