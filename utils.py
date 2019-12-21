import gym
import numpy as np
import torch


class RunningMean():
    """
    Utility class that tracks a running mean.
    """
    def __init__(self, n):
        self.list = np.empty(shape=(n))
        self.i = 0

    def mean(self):
        return np.nanmean(self.list)
    
    def __iadd__(self, other):
        self.list[self.i] = other
        self.i = (self.i + 1) % len(self.list)
        return self

    def __str__(self):
        return '{0:.4f}'.format(round(self.mean(), 4))

    def __len__(self):
        return len(self.list)


def transitions_to_sars(transitions):
    """
    Converts a list of transitions to individual SARS batches.
    """
    transition_list = list(zip(*transitions))
    contains_aux_info = len(transition_list) == 6
    if contains_aux_info:
        states, actions, rewards, new_states, dones, aux_info = transition_list
    else:
        states, actions, rewards, new_states, dones = transition_list 

    states = torch.stack(states)
    actions = torch.stack(actions).view(-1, 1)
    rewards = torch.stack([torch.Tensor([reward]) for reward in rewards])
    new_states = torch.stack(new_states)
    dones = torch.stack([torch.ByteTensor([done]) for done in dones])

    if contains_aux_info:
        return states, actions, rewards, new_states, dones, aux_info
    else: 
        return states, actions, rewards, new_states, dones
