# GoogleNetの実装

import torch
from torch import nn
import torch.nn.functional as F

from comm.nn import param_init

class InceptionModule(nn.Module):
    '''
    Inceptionモジュール

    branch1: conv1x1
    branch2: conv1x1 reduce -> conv3x3
    branch3: conv1x1 reduce -> conv5x5
    branch4: maxpool3x3 -> conv1x1
    '''

    def __init__(self, in_channels: int, 
                 branch1_out_channels: int, 
                 branch2_hidden_channels: int, branch2_out_channels: int,
                 branch3_hidden_channels: int, branch3_out_channels: int,
                 branch4_out_channels: int):
        super().__init__()

        self._branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch1_out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self._branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch2_hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch2_hidden_channels, branch2_out_channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
        )

        self._branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch3_hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch3_hidden_channels, branch3_out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        self._branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch4_out_channels, kernel_size=1), 
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        branch1_out = self._branch1(x)
        branch2_out = self._branch2(x)
        branch3_out = self._branch3(x)
        branch4_out = self._branch4(x)

        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

class InceptionV1(nn.Module):
    def __init__(self, classification_num: int):
        super().__init__()

        ########## Main ##########

        # 224x224x3 -> 112x112x64
        self._conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self._maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)

        # 112x112x64 -> 56x56x192
        self._conv2a = nn.Conv2d(64, 64, kernel_size=1)
        self._conv2b = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self._maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)

        # 56x56x192 -> 28x28x256
        self._inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)

        # 28x28x256 -> 28x28x480
        self._inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)

        # 28x28x480 -> 14x14x480
        self._maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 14x14x480 (4a)-> 14x14x512 (4b)-> 14x14x512 (4c)-> 14x14x512 (4d)-> 14x14x528 (4e)-> 14x14x832
        self._inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self._inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self._inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self._inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self._inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)

        # 14x14x832 -> 7x7x832
        self._maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 7x7x832 (5a)-> 7x7x832 (5b)-> 7x7x1024
        self._inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self._inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # avg pool: 7x7x1024 -> 1x1x1024
        self._avgpool_final = nn.AdaptiveAvgPool2d((1, 1))

        # fully connected layer: 1x1x1024 -> classification_num
        self._flatten_final = nn.Flatten()
        self._fc_final = nn.Linear(1024, classification_num)

        ########## Auxilary Classifier 1 ##########

        # 14x14x512 -> 4x4x512
        self._aux1_avgpool = nn.AvgPool2d(kernel_size=5, stride=3, padding=0)

        # 4x4x512 -> 4x4x128
        self._aux1_conv = nn.Conv2d(512, 128, kernel_size=1)

        # 4x4x128 -> 2048
        self._aux1_flatten = nn.Flatten()

        # 2048 -> 1024 -> classification_num
        self._aux1_fc1 = nn.Linear(2048, 1024)
        self._aux1_fc2 = nn.Linear(1024, classification_num)

        ########## Auxilary Classifier 2 ##########

        # 14x14x528 -> 4x4x528
        self._aux2_avgpool = nn.AvgPool2d(kernel_size=5, stride=3, padding=0)

        # 4x4x528 -> 4x4x128
        self._aux2_conv = nn.Conv2d(528, 128, kernel_size=1)

        # 4x4x128 -> 2048
        self._aux2_flatten = nn.Flatten()

        # 2048 -> 1024 -> classification_num
        self._aux2_fc1 = nn.Linear(2048, 1024)
        self._aux2_fc2 = nn.Linear(1024, classification_num)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        ##### layer sequence 1
        x = self._conv1(x)
        x = F.relu(x, inplace=True)
        x = self._maxpool1(x)
        x = self._lrn1(x)

        ##### layer sequence 2
        x = self._conv2a(x)
        x = F.relu(x, inplace=True)
        x = self._conv2b(x)
        x = F.relu(x, inplace=True)
        x = self._lrn2(x)
        x = self._maxpool2(x)

        ##### layer sequence 3 (inception3)
        x = self._inception3a(x)
        x = self._inception3b(x)
        x = self._maxpool3(x)

        ##### layer sequence 4 (inception4)
        x = self._inception4a(x)

        # auxiliary classifier 1
        aux1: torch.Tensor
        if self.training:
            aux1 = self._aux1_avgpool(x)
            aux1 = self._aux1_conv(aux1)
            aux1 = F.relu(aux1, inplace=True)
            aux1 = self._aux1_flatten(aux1)
            aux1 = self._aux1_fc1(aux1)
            aux1 = F.relu(aux1, inplace=True)
            aux1 = F.dropout(aux1, p=0.7, training=self.training)
            aux1 = self._aux1_fc2(aux1)
        else:
            aux1 = torch.zeros(1, self._aux1_fc2.out_features, device=x.device)

        x = self._inception4b(x)
        x = self._inception4c(x)
        x = self._inception4d(x)

        # auxiliary classifier 2
        aux2: torch.Tensor
        if self.training:
            aux2 = self._aux2_avgpool(x)
            aux2 = self._aux2_conv(aux2)
            aux2 = F.relu(aux2, inplace=True)
            aux2 = self._aux2_flatten(aux2)
            aux2 = self._aux2_fc1(aux2)
            aux2 = F.relu(aux2, inplace=True)
            aux2 = F.dropout(aux2, p=0.7, training=self.training)
            aux2 = self._aux2_fc2(aux2)
        else:
            aux2 = torch.zeros(1, self._aux2_fc2.out_features, device=x.device)

        x = self._inception4e(x)
        x = self._maxpool4(x)

        ##### layer sequence 5 (inception5)
        x = self._inception5a(x)
        x = self._inception5b(x)

        ##### final layer sequence (classification)
        x = self._avgpool_final(x)
        x = self._flatten_final(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self._fc_final(x)

        return x, aux1, aux2

######################################################################
#
#          BN-Inception (Inception v2)
#
#          chatGPT作
######################################################################

class _InceptionV2Conv(nn.Sequential):
    """Convolution followed by batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class _InceptionV2Block(nn.Module):
    """Four parallel branches at a fixed spatial resolution."""

    def __init__(
        self,
        in_channels: int,
        pointwise_channels: int,
        single_3x3_channels: tuple[int, int],
        double_3x3_channels: tuple[int, int, int],
        pool_channels: int,
        *,
        max_pool: bool = False,
    ):
        super().__init__()
        single_reduce, single_out = single_3x3_channels
        double_reduce, double_hidden, double_out = double_3x3_channels

        self._pointwise = _InceptionV2Conv(in_channels, pointwise_channels, 1)
        self._single_3x3 = nn.Sequential(
            _InceptionV2Conv(in_channels, single_reduce, 1),
            _InceptionV2Conv(single_reduce, single_out, 3),
        )
        self._double_3x3 = nn.Sequential(
            _InceptionV2Conv(in_channels, double_reduce, 1),
            _InceptionV2Conv(double_reduce, double_hidden, 3),
            _InceptionV2Conv(double_hidden, double_out, 3),
        )
        pool = (
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            if max_pool
            else nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        )
        self._pool_projection = nn.Sequential(
            pool,
            _InceptionV2Conv(in_channels, pool_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            self._pointwise(x),
            self._single_3x3(x),
            self._double_3x3(x),
            self._pool_projection(x),
        ), dim=1)

class _InceptionV2Reduction(nn.Module):
    """Downsample within each branch before concatenation."""

    def __init__(
        self,
        in_channels: int,
        single_3x3_channels: tuple[int, int],
        double_3x3_channels: tuple[int, int, int],
    ):
        super().__init__()
        single_reduce, single_out = single_3x3_channels
        double_reduce, double_hidden, double_out = double_3x3_channels

        self._single_3x3 = nn.Sequential(
            _InceptionV2Conv(in_channels, single_reduce, 1),
            _InceptionV2Conv(single_reduce, single_out, 3, stride=2),
        )
        self._double_3x3 = nn.Sequential(
            _InceptionV2Conv(in_channels, double_reduce, 1),
            _InceptionV2Conv(double_reduce, double_hidden, 3),
            _InceptionV2Conv(double_hidden, double_out, 3, stride=2),
        )
        self._pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self._single_3x3(x), self._double_3x3(x), self._pool(x)), dim=1)

class InceptionV2(nn.Module):
    """BN-Inception (Ioffe and Szegedy, 2015). Returns class logits of shape (N, C)."""

    def __init__(self, classification_num: int):
        super().__init__()

        # 224x224x3 -> 28x28x192. The first convolution is depthwise separable
        # with a depth multiplier of 8, followed by one BN and ReLU.
        self._stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=7, stride=2, padding=3, groups=3, bias=False),
            _InceptionV2Conv(24, 64, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _InceptionV2Conv(64, 64, 1),
            _InceptionV2Conv(64, 192, 3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self._inception = nn.Sequential(
            _InceptionV2Block(192, 64, (64, 64), (64, 96, 96), 32),        # 3a: 28x28x256
            _InceptionV2Block(256, 64, (64, 96), (64, 96, 96), 64),        # 3b: 28x28x320
            _InceptionV2Reduction(320, (128, 160), (64, 96, 96)),         # 3c: 14x14x576
            _InceptionV2Block(576, 224, (64, 96), (96, 128, 128), 128),    # 4a: 14x14x576
            _InceptionV2Block(576, 192, (96, 128), (96, 128, 128), 128),   # 4b: 14x14x576
            # TF-Slim uses 96-channel pool projections to match the 576-channel outputs.
            _InceptionV2Block(576, 160, (128, 160), (128, 160, 160), 96),  # 4c: 14x14x576
            _InceptionV2Block(576, 96, (128, 192), (160, 192, 192), 96),   # 4d: 14x14x576
            _InceptionV2Reduction(576, (128, 192), (192, 256, 256)),       # 4e: 7x7x1024
            _InceptionV2Block(1024, 352, (192, 320), (160, 224, 224), 128),  # 5a: 7x7x1024
            _InceptionV2Block(
                1024, 352, (192, 320), (192, 224, 224), 128, max_pool=True,
            ),  # 5b: 7x7x1024
        )
        self._global_pool = nn.AdaptiveAvgPool2d(1)
        self._classifier = nn.Linear(1024, classification_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._stem(x)
        x = self._inception(x)
        x = self._global_pool(x)
        return self._classifier(torch.flatten(x, 1))

######################################################################
#
#           Inception v3
#
######################################################################


class _InceptionV3Conv(nn.Sequential):
    """Convolution followed by batch normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True),
        )

        param_init(self)


class _InceptionV3Concat(nn.Module):
    def __init__(self, *branches: nn.Module):
        super().__init__()
        self._branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self._branches], dim=1)


def _inception_v3_a(in_channels: int, pool_channels: int) -> nn.Module:
    return _InceptionV3Concat(
        _InceptionV3Conv(in_channels, 64, 1),
        nn.Sequential(
            _InceptionV3Conv(in_channels, 48, 1),
            _InceptionV3Conv(48, 64, 5, padding=2),
        ),
        nn.Sequential(
            _InceptionV3Conv(in_channels, 64, 1),
            _InceptionV3Conv(64, 96, 3, padding=1),
            _InceptionV3Conv(96, 96, 3, padding=1),
        ),
        nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            _InceptionV3Conv(in_channels, pool_channels, 1),
        ),
    )


def _inception_v3_reduction_a() -> nn.Module:
    return _InceptionV3Concat(
        _InceptionV3Conv(288, 384, 3, stride=2),
        nn.Sequential(
            _InceptionV3Conv(288, 64, 1),
            _InceptionV3Conv(64, 96, 3, padding=1),
            _InceptionV3Conv(96, 96, 3, stride=2),
        ),
        nn.MaxPool2d(3, stride=2),
    )


def _inception_v3_b(channels_7x7: int) -> nn.Module:
    return _InceptionV3Concat(
        _InceptionV3Conv(768, 192, 1),
        nn.Sequential(
            _InceptionV3Conv(768, channels_7x7, 1),
            _InceptionV3Conv(channels_7x7, channels_7x7, (1, 7), padding=(0, 3)),
            _InceptionV3Conv(channels_7x7, 192, (7, 1), padding=(3, 0)),
        ),
        nn.Sequential(
            _InceptionV3Conv(768, channels_7x7, 1),
            _InceptionV3Conv(channels_7x7, channels_7x7, (7, 1), padding=(3, 0)),
            _InceptionV3Conv(channels_7x7, channels_7x7, (1, 7), padding=(0, 3)),
            _InceptionV3Conv(channels_7x7, channels_7x7, (7, 1), padding=(3, 0)),
            _InceptionV3Conv(channels_7x7, 192, (1, 7), padding=(0, 3)),
        ),
        nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            _InceptionV3Conv(768, 192, 1),
        ),
    )


def _inception_v3_reduction_b() -> nn.Module:
    return _InceptionV3Concat(
        nn.Sequential(
            _InceptionV3Conv(768, 192, 1),
            _InceptionV3Conv(192, 320, 3, stride=2),
        ),
        nn.Sequential(
            _InceptionV3Conv(768, 192, 1),
            _InceptionV3Conv(192, 192, (1, 7), padding=(0, 3)),
            _InceptionV3Conv(192, 192, (7, 1), padding=(3, 0)),
            _InceptionV3Conv(192, 192, 3, stride=2),
        ),
        nn.MaxPool2d(3, stride=2),
    )


def _inception_v3_c(in_channels: int, pool: nn.Module) -> nn.Module:
    return _InceptionV3Concat(
        _InceptionV3Conv(in_channels, 320, 1),
        nn.Sequential(
            _InceptionV3Conv(in_channels, 384, 1),
            _InceptionV3Concat(
                _InceptionV3Conv(384, 384, (1, 3), padding=(0, 1)),
                _InceptionV3Conv(384, 384, (3, 1), padding=(1, 0)),
            ),
        ),
        nn.Sequential(
            _InceptionV3Conv(in_channels, 448, 1),
            _InceptionV3Conv(448, 384, 3, padding=1),
            _InceptionV3Concat(
                _InceptionV3Conv(384, 384, (1, 3), padding=(0, 1)),
                _InceptionV3Conv(384, 384, (3, 1), padding=(1, 0)),
            ),
        ),
        nn.Sequential(pool, _InceptionV3Conv(in_channels, 192, 1)),
    )


class _InceptionV3Aux(nn.Module):
    def __init__(self, classification_num: int):
        super().__init__()
        self._projection = _InceptionV3Conv(768, 128, 1)
        self._bottleneck = _InceptionV3Conv(128, 768, 5)
        self._classifier = nn.Linear(768, classification_num)

        param_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve the supplementary pooling on 17x17 features from 299x299
        # inputs. Other resolutions use a 5x5 grid for the same bottleneck.
        if x.shape[-2:] == (17, 17):
            x = F.avg_pool2d(x, kernel_size=5, stride=3)
        else:
            x = F.adaptive_avg_pool2d(x, (5, 5))
        x = self._projection(x)
        x = self._bottleneck(x)
        return self._classifier(torch.flatten(x, 1))


class InceptionV3(nn.Module):
    """Inception-v3 based on arXiv:1512.00567v3 supplementary model.txt.

    Returns (main_logits, aux_logits). The auxiliary output is available only
    in training mode with aux_logits=True, which requires at least two
    examples per batch for the auxiliary bottleneck's batch normalization.
    """

    def __init__(self, classification_num: int, *, aux_logits: bool = True):
        super().__init__()
        self._stem = nn.Sequential(
            _InceptionV3Conv(3, 32, 3, stride=2),
            _InceptionV3Conv(32, 32, 3),
            _InceptionV3Conv(32, 64, 3, padding=1),
            nn.MaxPool2d(3, stride=2),
            _InceptionV3Conv(64, 80, 1),
            _InceptionV3Conv(80, 192, 3),
            nn.MaxPool2d(3, stride=2),
        )

        self._stage_35 = nn.Sequential(
            _inception_v3_a(192, 32),  # mixed:   35x35x256
            _inception_v3_a(256, 64),  # mixed_1: 35x35x288
            _inception_v3_a(288, 64),  # mixed_2: 35x35x288
        )
        self._reduce_35 = _inception_v3_reduction_a()  # mixed_3: 17x17x768
        self._stage_17 = nn.Sequential(
            _inception_v3_b(128),  # mixed_4
            _inception_v3_b(160),  # mixed_5
            _inception_v3_b(160),  # mixed_6
            _inception_v3_b(192),  # mixed_7
        )
        self._aux = _InceptionV3Aux(classification_num) if aux_logits else None
        self._reduce_17 = _inception_v3_reduction_b()  # mixed_8: 8x8x1280
        self._stage_8 = nn.Sequential(
            _inception_v3_c(
                1280, nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            ),  # mixed_9: 8x8x2048
            _inception_v3_c(
                2048, nn.MaxPool2d(3, stride=1, padding=1),
            ),  # mixed_10: 8x8x2048
        )
        self._global_pool = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(p=0.2)
        self._classifier = nn.Linear(2048, classification_num)

        param_init(self)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._stage_35(self._stem(x))
        x = self._stage_17(self._reduce_35(x))
        aux = self._aux(x) if self.training and self._aux is not None else torch.zeros(1, self._classifier.out_features, device=x.device)
        x = self._stage_8(self._reduce_17(x))
        x = torch.flatten(self._global_pool(x), 1)
        return self._classifier(self._dropout(x)), aux
