import random


class CircularBuffer(object):
    """
    Implements a simple circular buffer. Does not support pushing of slices.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []
        self.pointer = 0
        self.n_items = 0

    def push(self, *args):
        if len(self.items) < self.capacity:
            self.items.append(None)
        self.items[self.pointer] = args
        self.pointer = (self.pointer + 1) % self.capacity
        self.n_items = min(self.n_items + 1, self.capacity)

    def __getitem__(self, key):
        return self.items[key]

    def __iadd__(self, args):
        self.push(*args)
        return self

    def __len__(self):
        assert self.n_items == len(self.items)
        return self.n_items


class ReplayMemory(CircularBuffer):
    """
    Implements a simple replay memory that we can sample from.
    """
    def __init__(self, capacity):
        super().__init__(capacity)

    def sample(self, batch_size):
        return random.sample(self.items, batch_size)
