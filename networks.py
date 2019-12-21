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
