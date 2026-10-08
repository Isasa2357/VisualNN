# ResNetの実装

import torch
from torch import nn
import torch.nn.functional as F

from comm.nn import param_init

class BasicBlock(nn.Module):
    '''
    ResNetのBasicBlock．2つの畳み込みとショートカット機構を備えている
    入力チャネルと出力チャネルを変更できるほか，reduceをTrueにすると空間解像度を半分にすることができる
    '''
    def __init__(self, in_channels: int, out_channels: int, reduce: bool = False):
        super().__init__()

        conv1_stride = 2 if reduce else 1
        self._conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=conv1_stride, padding=1, bias=False)
        self._bn1 = nn.BatchNorm2d(out_channels)

        self._conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn2 = nn.BatchNorm2d(out_channels)

        self._shortcut = nn.Sequential()
        if reduce or in_channels != out_channels:
            shortcut_stride = 2 if reduce else 1
            self._shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=shortcut_stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self._bn1(self._conv1(x)))
        out = self._bn2(self._conv2(out))
        out += self._shortcut(x)
        out = F.relu(out)
        return out

class BottleneckBlock(nn.Module):
    '''
    ResNetのBottleneckBlock．3つの畳み込みとショートカット機構を備えている
    入力チャネルと出力チャネルを変更できるほか，reduceをTrueにすると空間解像度を半分にすることができる
    '''

    def __init__(self, in_channels: int, out_channels: int, inner_channels: int, reduce: bool = False):
        super().__init__()

        conv1_stride = 2 if reduce else 1
        self._conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=1, bias=False)
        self._bn1 = nn.BatchNorm2d(inner_channels)

        self._conv2 = nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=conv1_stride, padding=1, bias=False)
        self._bn2 = nn.BatchNorm2d(inner_channels)

        self._conv3 = nn.Conv2d(inner_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self._bn3 = nn.BatchNorm2d(out_channels)

        self._shortcut = nn.Sequential()
        if reduce or in_channels != out_channels:
            shortcut_stride = 2 if reduce else 1
            self._shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=shortcut_stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self._bn1(self._conv1(x)))
        out = F.relu(self._bn2(self._conv2(out)))
        out = self._bn3(self._conv3(out))
        out += self._shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()

        # conv1: 224x224x3 -> 112x112x64
        self._conv1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), 
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )

        # conv2: 112x112x64 -> 56x56x64
        self._conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(64, 64, reduce=False),
            BasicBlock(64, 64, reduce=False)
        )

        # conv3: 56x56x64 -> 28x28x128
        self._conv3_x = nn.Sequential(
            BasicBlock(64, 128, reduce=True),
            BasicBlock(128, 128, reduce=False)
        )

        # conv4: 28x28x128 -> 14x14x256
        self._conv4_x = nn.Sequential(
            BasicBlock(128, 256, reduce=True),
            BasicBlock(256, 256, reduce=False)
        )

        # conv5: 14x14x256 -> 7x7x512
        self._conv5_x = nn.Sequential(
            BasicBlock(256, 512, reduce=True),
            BasicBlock(512, 512, reduce=False)
        )

        self._avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._fc = nn.Linear(512, num_classes)

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1(x)
        x = self._conv2_x(x)
        x = self._conv3_x(x)
        x = self._conv4_x(x)
        x = self._conv5_x(x)
        x = self._avgpool(x)
        x = torch.flatten(x, 1)
        x = self._fc(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # conv1: 224x224x3 -> 112x112x64
        self._conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # conv2: 112x112x64 -> 56x56x64
        self._conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(64, 64, reduce=False),
            BasicBlock(64, 64, reduce=False), 
            BasicBlock(64, 64, reduce=False)
        )

        # conv3: 56x56x64 -> 28x28x128
        self._conv3_x = nn.Sequential(
            BasicBlock(64, 128, reduce=True),
            BasicBlock(128, 128, reduce=False), 
            BasicBlock(128, 128, reduce=False),
            BasicBlock(128, 128, reduce=False)
        )

        # conv4: 28x28x128 -> 14x14x256
        self._conv4_x = nn.Sequential(
            BasicBlock(128, 256, reduce=True),
            BasicBlock(256, 256, reduce=False),
            BasicBlock(256, 256, reduce=False),
            BasicBlock(256, 256, reduce=False), 
            BasicBlock(256, 256, reduce=False), 
            BasicBlock(256, 256, reduce=False)
        )

        # conv5: 14x14x256 -> 7x7x512
        self._conv5_x = nn.Sequential(
            BasicBlock(256, 512, reduce=True),
            BasicBlock(512, 512, reduce=False), 
            BasicBlock(512, 512, reduce=False)
        )

        self._avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._fc = nn.Linear(512, num_classes)

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1(x)
        x = self._conv2_x(x)
        x = self._conv3_x(x)
        x = self._conv4_x(x)
        x = self._conv5_x(x)
        x = self._avgpool(x)
        x = torch.flatten(x, 1)
        x = self._fc(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # conv1: 224x224x3 -> 112x112x64
        self._conv1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), 
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )

        # conv2: 112x112x64 -> 56x56x256
        self._conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BottleneckBlock(64, 256, 64, reduce=False),
            BottleneckBlock(256, 256, 64, reduce=False), 
            BottleneckBlock(256, 256, 64, reduce=False)
        )

        # conv3: 56x56x256 -> 28x28x512
        self._conv3_x = nn.Sequential(
            BottleneckBlock(256, 512, 128, reduce=True),
            BottleneckBlock(512, 512, 128, reduce=False),
            BottleneckBlock(512, 512, 128, reduce=False), 
            BottleneckBlock(512, 512, 128, reduce=False)
        )

        # conv4: 28x28x512 -> 14x14x1024
        self._conv4_x = nn.Sequential(
            BottleneckBlock(512, 1024, 256, reduce=True),
            BottleneckBlock(1024, 1024, 256, reduce=False),
            BottleneckBlock(1024, 1024, 256, reduce=False),
            BottleneckBlock(1024, 1024, 256, reduce=False), 
            BottleneckBlock(1024, 1024, 256, reduce=False), 
            BottleneckBlock(1024, 1024, 256, reduce=False)
        )

        # conv5: 14x14x1024 -> 7x7x2048
        self._conv5_x = nn.Sequential(
            BottleneckBlock(1024, 2048, 512, reduce=True),
            BottleneckBlock(2048, 2048, 512, reduce=False),
            BottleneckBlock(2048, 2048, 512, reduce=False)
        )

        self._avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._fc = nn.Linear(2048, num_classes)

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1(x)
        x = self._conv2_x(x)
        x = self._conv3_x(x)
        x = self._conv4_x(x)
        x = self._conv5_x(x)
        x = self._avgpool(x)
        x = torch.flatten(x, 1)
        x = self._fc(x)

        return x

class PreActivationBasicBlock(nn.Module):
    '''
    pre activationを採用したBasicBlock．
    '''
    def __init__(self, in_channels: int, out_channels: int, reduce: bool = False):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._reduce = reduce

        conv1_stride = 2 if reduce else 1

        self._comm_layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 残差関数
        # リサイズする場合，conv1で変更する
        # conv後にそのままBNに入らないため，biasはTrue
        self._preact_residual_func = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=conv1_stride, padding=1, bias=True), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        )

        # ショートカット関数
        self._shortcut = nn.Sequential()
        if reduce or in_channels != out_channels:
            self._shortcut = nn.Sequential(
                nn.Conv2d(self._in_channels, self._out_channels, kernel_size=1, stride=2 if reduce else 1, padding=0, bias=True)
            )

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self._comm_layers(x)

        out = self._preact_residual_func(x)

        shortcut_out: torch.Tensor
        if self._reduce or self._in_channels != self._out_channels:
            shortcut_out = self._shortcut(x)
        else:
            shortcut_out = identity

        return out + shortcut_out

class PreActivationBottleneckBlock(nn.Module):
    '''
    pre activationを採用したBottleneckBlock．
    '''
    def __init__(self, in_channels: int, out_channels: int, inner_channels: int, reduce: bool = False):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._inner_channels = inner_channels
        self._reduce = reduce

        # ショートカット時の共通レイヤ
        self._common_layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 残差関数
        self._preact_residual_func = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            # ダウンサンプルはこのレイヤで行う
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2 if reduce else 1, padding=1, bias=True),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # ショートカット
        self._shortcut = nn.Sequential()
        if reduce or in_channels != out_channels:
            self._shortcut = nn.Sequential(
                nn.Conv2d(self._in_channels, self._out_channels, kernel_size=1, stride=2 if reduce else 1, padding=0, bias=True)
            )

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 入力はバッチ付き2.5次元テンソルかつ入力チャネルが正しいことを判定
        if x.ndim != 4 or x.size(1) != self._in_channels:
            raise ValueError(f"Input tensor must have shape (batch_size, {self._in_channels}, height, width), but got {x.shape}")

        identity = x
        x = self._common_layers(x)
        out = self._preact_residual_func(x)

        shortcut_out: torch.Tensor
        if self._reduce or self._in_channels != self._out_channels:
            shortcut_out = self._shortcut(x)
        else:
            shortcut_out = identity

        return out + shortcut_out

class PreactivationResNet18(nn.Module):
    '''
    preactivation版ResNet18．
    '''

    def __init__(self, num_classes: int):
        super().__init__()

        # conv1: 224x224x3 -> 112x112x64
        self._conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        )

        # conv2_x: 112x112x64 -> 56x56x64
        self._conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            PreActivationBasicBlock(64, 64, reduce=False),
            PreActivationBasicBlock(64, 64, reduce=False)
        )

        # conv3_x: 56x56x64 -> 28x28x128
        self._conv3_x = nn.Sequential(
            PreActivationBasicBlock(64, 128, reduce=True),
            PreActivationBasicBlock(128, 128, reduce=False)
        )

        # conv4_x: 28x28x128 -> 14x14x256
        self._conv4_x = nn.Sequential(
            PreActivationBasicBlock(128, 256, reduce=True),
            PreActivationBasicBlock(256, 256, reduce=False)
        )

        # conv5_x: 14x14x256 -> 7x7x512
        self._conv5_x = nn.Sequential(
            PreActivationBasicBlock(256, 512, reduce=True),
            PreActivationBasicBlock(512, 512, reduce=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self._final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1(x)
        x = self._conv2_x(x)
        x = self._conv3_x(x)
        x = self._conv4_x(x)
        x = self._conv5_x(x)
        x = self._final_layers(x)

        return x

class PreactivationResNet50(nn.Module):
    '''
    preactivation版ResNet50．
    '''

    def __init__(self, num_classes: int):
        super().__init__()

        # conv1: 224x224x3 -> 112x112x64
        self._conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        )

        # conv2_x: 112x112x64 -> 56x56x256
        self._conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            PreActivationBottleneckBlock(64, 256, 64, reduce=False),
            PreActivationBottleneckBlock(256, 256, 64, reduce=False),
            PreActivationBottleneckBlock(256, 256, 64, reduce=False)
        )

        # conv3_x: 56x56x256 -> 28x28x512
        self._conv3_x = nn.Sequential(
            PreActivationBottleneckBlock(256, 512, 128, reduce=True),
            PreActivationBottleneckBlock(512, 512, 128, reduce=False),
            PreActivationBottleneckBlock(512, 512, 128, reduce=False),
            PreActivationBottleneckBlock(512, 512, 128, reduce=False)
        )

        # conv4_x: 28x28x512 -> 14x14x1024
        self._conv4_x = nn.Sequential(
            PreActivationBottleneckBlock(512, 1024, 256, reduce=True),
            PreActivationBottleneckBlock(1024, 1024, 256, reduce=False),
            PreActivationBottleneckBlock(1024, 1024, 256, reduce=False),
            PreActivationBottleneckBlock(1024, 1024, 256, reduce=False),
            PreActivationBottleneckBlock(1024, 1024, 256, reduce=False),
            PreActivationBottleneckBlock(1024, 1024, 256, reduce=False)
        )

        # conv5_x: 14x14x1024 -> 7x7x2048
        self._conv5_x = nn.Sequential(
            PreActivationBottleneckBlock(1024, 2048, 512, reduce=True),
            PreActivationBottleneckBlock(2048, 2048, 512, reduce=False),
            PreActivationBottleneckBlock(2048, 2048, 512, reduce=False), 
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self._final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1(x)
        x = self._conv2_x(x)
        x = self._conv3_x(x)
        x = self._conv4_x(x)
        x = self._conv5_x(x)
        x = self._final_layers(x)

        return x

class PreactivationWideResNetBasicBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, depth: int, reduce: bool, drop_rate: float = 0.0):
        super().__init__()

        # 入力確認
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if drop_rate < 0.0 or drop_rate > 1.0:
            raise ValueError("drop_rate must be between 0.0 and 1.0")

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._depth = depth
        self._reduce = reduce
        self._drop_rate = drop_rate

        self._comm_layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        layers_lst: nn.ModuleList = nn.ModuleList()
        shortcut_lst: nn.ModuleList = nn.ModuleList()

        for i in range(depth):
            if i == 0:
                # ダウンサンプルの場合，畳み込み層で行う
                layers_lst.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if reduce else 1, padding=1, bias=False))
            else:
                layers_lst.append(nn.BatchNorm2d(out_channels))
                layers_lst.append(nn.ReLU(inplace=True))
                if i == depth - 1 and self._drop_rate > 0.0:
                    layers_lst.append(nn.Dropout(p=self._drop_rate))
                layers_lst.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        if reduce or in_channels != out_channels:
            shortcut_lst.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if reduce else 1, bias=False))

        self._layers = nn.Sequential(*layers_lst)
        self._shortcut = nn.Sequential(*shortcut_lst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self._comm_layers(x)
        out = self._layers(x)

        if self._shortcut:
            out += self._shortcut(x)
        else:
            out += identity

        return out

class PostactivationWideResNetBasicBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, depth: int, reduce: bool = False, drop_rate: float = 0.0):
        super().__init__()

        # 入力の確認
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if drop_rate < 0.0 or drop_rate > 1.0:
            raise ValueError("drop_rate must be between 0.0 and 1.0")

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._depth = depth
        self._reduce = reduce
        self._drop_rate = drop_rate

        layers_lst: nn.ModuleList = nn.ModuleList()
        shortcut_lst: nn.ModuleList = nn.ModuleList()

        for i in range(depth):
            if i == 0:
                # 最初の層．ダウンサンプルを行う場合がある
                layers_lst.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if reduce else 1, padding=1, bias=False))
                layers_lst.append(nn.BatchNorm2d(out_channels))
                layers_lst.append(nn.ReLU(inplace=True))
            elif i == depth - 1:
                # 最後の層．活性化はしない．ドロップアウトを行う場合がある
                if drop_rate > 0.0:
                    layers_lst.append(nn.Dropout(p=drop_rate))
                layers_lst.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
                layers_lst.append(nn.BatchNorm2d(out_channels))
            else:
                # 中間の層
                layers_lst.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
                layers_lst.append(nn.BatchNorm2d(out_channels))
                layers_lst.append(nn.ReLU(inplace=True))

        if reduce or in_channels != out_channels:
            shortcut_lst.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if reduce else 1, padding=0, bias=False))
            shortcut_lst.append(nn.BatchNorm2d(out_channels))

        self._layers = nn.Sequential(*layers_lst)
        self._shortcut = nn.Sequential(*shortcut_lst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # return F.relu(self._layers(x) + self._shortcut(x))
        # print("main")
        out = self._layers(x)
        # print("shortcut")
        shortcut = self._shortcut(x)
        # print(f"in block: out shape = {out.shape}, shortcut shape = {shortcut.shape}")
        out += shortcut
        return F.relu(out)

class WideResNet18(nn.Module):

    '''
    Wide ResNetの実装
    imagenet(224x224)向けの実装
    '''

    def __init__(self, num_classes: int, widen_factor: float, drop_rate: float = 0.0, preactivation: bool = False):
        super().__init__()

        self._num_classes = num_classes
        self._block_depth = 2
        self._widen_factor = widen_factor
        self._drop_rate = drop_rate
        self._preactivation = preactivation

        # conv1
        in_channels = 3
        out_channels = int(64 * widen_factor)
        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
        )
        if not preactivation:
            self._conv1.append(nn.BatchNorm2d(out_channels))
            self._conv1.append(nn.ReLU(inplace=True))

        # conv2
        in_channels = out_channels
        out_channels = int(64 * widen_factor)
        self._conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 
            PreactivationWideResNetBasicBlock(in_channels, out_channels, self._block_depth, False, drop_rate) 
                if preactivation else PostactivationWideResNetBasicBlock(in_channels, out_channels, self._block_depth, False, drop_rate), 
            PreactivationWideResNetBasicBlock(in_channels, out_channels, self._block_depth, False, drop_rate) 
                if preactivation else PostactivationWideResNetBasicBlock(in_channels, out_channels, self._block_depth, False, drop_rate)
        )

        # conv3
        in_channels = out_channels
        out_channels = int(128 * widen_factor)
        self._conv3 = nn.Sequential(
            PreactivationWideResNetBasicBlock(in_channels, out_channels, self._block_depth, True, drop_rate) 
                if preactivation else PostactivationWideResNetBasicBlock(in_channels, out_channels, self._block_depth, True, drop_rate), 
            PreactivationWideResNetBasicBlock(out_channels, out_channels, self._block_depth, False, drop_rate) 
                if preactivation else PostactivationWideResNetBasicBlock(out_channels, out_channels, self._block_depth, False, drop_rate)
        )

        # conv4
        in_channels = out_channels
        out_channels = int(256 * widen_factor)
        self._conv4 = nn.Sequential(
            PreactivationWideResNetBasicBlock(in_channels, out_channels, self._block_depth, True, drop_rate) 
                if preactivation else PostactivationWideResNetBasicBlock(in_channels, out_channels, self._block_depth, True, drop_rate), 
            PreactivationWideResNetBasicBlock(out_channels, out_channels, self._block_depth, False, drop_rate) 
                if preactivation else PostactivationWideResNetBasicBlock(out_channels, out_channels, self._block_depth, False, drop_rate)
        )

        # conv5
        in_channels = out_channels
        out_channels = int(512 * widen_factor)
        self._conv5 = nn.Sequential(
            PreactivationWideResNetBasicBlock(in_channels, out_channels, self._block_depth, True, drop_rate) 
                if preactivation else PostactivationWideResNetBasicBlock(in_channels, out_channels, self._block_depth, True, drop_rate), 
            PreactivationWideResNetBasicBlock(out_channels, out_channels, self._block_depth, False, drop_rate) 
                if preactivation else PostactivationWideResNetBasicBlock(out_channels, out_channels, self._block_depth, False, drop_rate), 
        )
        if preactivation:
            self._conv5.append(nn.BatchNorm2d(out_channels))
            self._conv5.append(nn.ReLU(inplace=True))

        # classifier
        self._classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes)
        )

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x = self._conv1(x)
        # print(x.shape)
        x = self._conv2(x)
        # print(x.shape)
        x = self._conv3(x)
        # print(x.shape)
        x = self._conv4(x)
        # print(x.shape)
        x = self._conv5(x)
        # print(x.shape)
        x = self._classifier(x)
        return x
