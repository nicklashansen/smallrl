import numpy as np

class LambdaEpsilonSchedule:
    def __init__(self, func):
        self.func = func

    def __getitem__(self, key):
        assert key >= 0
        return self.func(key)


class LinearEpsilonSchedule(LambdaEpsilonSchedule):
    def __init__(self, eps_start, eps_end, num_steps):
        func = lambda step: eps_end if step >= num_steps else eps_start + step * (eps_end - eps_start) / num_steps
        super().__init__(func)
