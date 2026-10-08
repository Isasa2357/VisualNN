import torch
import torch.nn as nn
import torch.nn.functional as F

def param_init(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list[int], output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim[0]))
        for i in range(1, len(hidden_dim)):
            self.layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
        self.layers.append(nn.Linear(hidden_dim[-1], output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

