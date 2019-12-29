import torch
import torch.nn as nn


class Agent(nn.Module):

    def __init__(self, env, net, optimizer):
        super().__init__()
        self.env = env
        self.net = net
        self.optimizer = optimizer
        self.algorithm = self.__class__.__name__

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def process_state(self, state):
        """
        Can be overwritten to process observed states by e.g. encoding.
        """
        return state

    def reset(self):
        """
        Resets the current environment.
        """
        return self.process_state(self.env.reset())

    def step(self):
        """
        Takes a step in the environment.
        """
        raise NotImplementedError()

    def generate_rollout(self):
        """
        Generates a rollout using the current policy.
        """
        raise NotImplementedError()

    def compute_loss(self):
        """
        Computes the loss from a given rollout.
        """
        raise NotImplementedError()
    
    def optimize(self, loss):
        """
        Optimizes the current policy given a predefined optimizer and computed loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
