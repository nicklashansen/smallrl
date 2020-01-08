import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """
    Multi-layer perceptron network.
    """
    def __init__(self, *sizes):
        super().__init__()
        assert len(sizes) > 1
        layers = []
        
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(sizes[i-1], sizes[i]))
            if i != len(sizes)-1: layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)


class TwoHeadedMLP(nn.Module):
    """
    Two-headed multi-layer perceptron network for Actor-Critic algorithms.
    """
    def __init__(self, *sizes):
        super().__init__()
        self.encoder = MLP(*sizes[:-1])
        self.policy = nn.Linear(sizes[-2], sizes[-1])
        self.value = nn.Linear(sizes[-2], 1)

    def forward(self, x):
        enc = F.relu(self.encoder(x))
        return self.policy(enc), self.value(enc)


class CNN(nn.Module):
    """
    Convolutional neural network.
    """
    def __init__(self, channels, kernel_sizes, paddings):
        super().__init__()
        assert len(channels) > 1
        assert len(kernel_sizes) == len(channels)-1
        assert len(paddings) == len(channels)-1
        layers = []
        
        for i in range(1, len(channels)):
            layers.append(nn.Conv2d(
                in_channels=channels[i-1],
                out_channels=channels[i],
                kernel_size=kernel_sizes[i-1],
                padding=paddings[i-1])
            )
            if i != len(channels)-1: layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CNN_MLP(nn.Module):
    """
    Convolutional neural network with a fully connected output.
    """
    def __init__(self, channels, kernel_sizes, paddings, sizes):
        super().__init__()
        self.cnn = CNN(channels, kernel_sizes, paddings)
        self.mlp = MLP(*sizes)


    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        
        return self.mlp(x)
