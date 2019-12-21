import torch
import torch.nn.functional as F
import numpy as np


def sample_logits(logits):
    """
    Samples from a categorical distribution with given logits.
    """
    return torch.distributions.Categorical(logits=logits).sample()


def entropy(probs):
    """
    Computes the entropy of a normalized probability distribution.
    """   
    return -(torch.log(probs) * probs).sum(dim=-1)


def entropy_logits(logits):
    """
    Computes the entropy of an unnormalized probability distribution.
    """
    probs = F.softmax(logits, dim=-1)
    return entropy(probs)
    