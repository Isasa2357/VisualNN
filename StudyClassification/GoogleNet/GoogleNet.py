# GoogleNetの実装

import torch
from torch import nn
import torch.nn.functional as F

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

        self._branch1 = nn.Conv2d(in_channels, branch1_out_channels, kernel_size=1)

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

class GoogleNet(nn.Module):
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
        aux1 = self._aux1_avgpool(x)
        aux1 = self._aux1_conv(aux1)
        aux1 = F.relu(aux1, inplace=True)
        aux1 = self._aux1_flatten(aux1)
        aux1 = self._aux1_fc1(aux1)
        aux1 = F.relu(aux1, inplace=True)
        aux1 = self._aux1_fc2(aux1)

        x = self._inception4b(x)
        x = self._inception4c(x)
        x = self._inception4d(x)

        # auxiliary classifier 2
        aux2 = self._aux2_avgpool(x)
        aux2 = self._aux2_conv(aux2)
        aux2 = F.relu(aux2, inplace=True)
        aux2 = self._aux2_flatten(aux2)
        aux2 = self._aux2_fc1(aux2)
        aux2 = F.relu(aux2, inplace=True)
        aux2 = self._aux2_fc2(aux2)

        x = self._inception4e(x)
        x = self._maxpool4(x)

        ##### layer sequence 5 (inception5)
        x = self._inception5a(x)
        x = self._inception5b(x)

        ##### final layer sequence (classification)
        x = self._avgpool_final(x)
        x = self._flatten_final(x)
        x = self._fc_final(x)

        return x, aux1, aux2