"""ImageNet100向けResNet18。se引数でSE-Blockの有無を比較する。"""

import torch
from torch import nn


class SEBlock(nn.Module):
    """空間方向の平均からチャネルごとの重みを学習する。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("channels and reduction must be positive")

        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.excitation(self.pool(x))


class BasicBlock(nn.Module):
    """SEは残差枝の最後のBatchNormの後、ショートカット加算の前に適用。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        se: bool = False,
        reduction: int = 16,
    ):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SEBlock(out_channels, reduction) if se else nn.Identity()
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.se(self.residual(x)) + self.shortcut(x))


class NN(nn.Module):
    """ImageNet用のstemを持つResNet18（各stageにBasicBlockを2個）。

    Args:
        num_classes: 分類クラス数。ImageNet100では100。
        se: Trueなら全8個の残差ブロックにSEを追加する。
        reduction: SEの中間チャネル数の縮小率。

    入力: (N, 3, H, W)。標準の入力解像度は224 x 224。
    出力: (N, num_classes)のlogits。CrossEntropyLossに直接渡せる。

    Example:
        baseline = NN(num_classes=100, se=False)
        senet = NN(num_classes=100, se=True)
    """

    def __init__(self, num_classes: int, se: bool = False, reduction: int = 16):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not isinstance(se, bool):
            raise TypeError("se must be bool")
        if reduction <= 0:
            raise ValueError("reduction must be positive")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stages = nn.Sequential(
            BasicBlock(64, 64, se=se, reduction=reduction),
            BasicBlock(64, 64, se=se, reduction=reduction),
            BasicBlock(64, 128, stride=2, se=se, reduction=reduction),
            BasicBlock(128, 128, se=se, reduction=reduction),
            BasicBlock(128, 256, stride=2, se=se, reduction=reduction),
            BasicBlock(256, 256, se=se, reduction=reduction),
            BasicBlock(256, 512, stride=2, se=se, reduction=reduction),
            BasicBlock(512, 512, se=se, reduction=reduction),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stages(self.stem(x)))
