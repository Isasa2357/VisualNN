# DenseNet network implementation in PyTorch

import torch
from torch import nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()

        hidden_channels = int(in_channels / 2)

        self._layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(hidden_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)

class DenseBlock(nn.Module):
    '''
    複数のDenseLayerから構成されるブロック
    '''

    def __init__(self, num_layers: int, in_channels: int, growth_rate: int):
        '''
        num_layers: DenseBlockを構成するDenseLayerの数
        in_channels: 入力チャネル数
        growth_rate: 各DenseLayerの成長率
        '''
        super().__init__()

        self._numlayers = num_layers
        self._growth_rate = growth_rate

        self._layers = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self._layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x

class TransitionLayer(nn.Module):
    '''
    DenseNetのTransitionLayer
    DenseBlockで肥大化したチャネル数を圧縮する
    '''

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self._layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)

class DenseNet121(nn.Module):

    def __init__(self, num_classes: int, growth_rate: int):
        super().__init__()

        self._initial_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # dense layer + transition layer 1
        inchnl = 64
        dense_outchnl = inchnl + 6 * growth_rate
        transition_outchnl = dense_outchnl // 2
        self._dense_block1 = DenseBlock(num_layers=6, in_channels=inchnl, growth_rate=growth_rate)
        self._transition1 = TransitionLayer(in_channels=dense_outchnl, out_channels=transition_outchnl)

        # dense layer + transition layer 2
        inchnl = transition_outchnl
        dense_outchnl = inchnl + 12 * growth_rate
        transition_outchnl = dense_outchnl // 2
        self._dense_block2 = DenseBlock(num_layers=12, in_channels=inchnl, growth_rate=growth_rate)
        self._transition2 = TransitionLayer(in_channels=dense_outchnl, out_channels=transition_outchnl)

        # dense layer + transition layer 3
        inchnl = transition_outchnl
        dense_outchnl = inchnl + 24 * growth_rate
        transition_outchnl = dense_outchnl // 2
        self._dense_block3 = DenseBlock(num_layers=24, in_channels=inchnl, growth_rate=growth_rate)
        self._transition3 = TransitionLayer(in_channels=dense_outchnl, out_channels=transition_outchnl)

        # dense layer 4 (no transition layer after this)
        inchnl = transition_outchnl
        dense_outchnl = inchnl + 16 * growth_rate
        self._dense_block4 = DenseBlock(num_layers=16, in_channels=inchnl, growth_rate=growth_rate)

        self._final_layers = nn.Sequential(
            nn.BatchNorm2d(dense_outchnl),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dense_outchnl, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError("Input tensor must have shape (batch_size, 3, height, width). but got shape {}".format(x.shape))

        x = self._initial_layers(x)
        x = self._dense_block1(x)
        x = self._transition1(x)
        x = self._dense_block2(x)
        x = self._transition2(x)
        x = self._dense_block3(x)
        x = self._transition3(x)
        x = self._dense_block4(x)
        x = self._final_layers(x)

        return x