import torch
import gym


class TorchEnv(gym.Wrapper):
    """
    Wrapper environment for interacting with a PyTorch agent.
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        return torch.FloatTensor(self.env.reset(**kwargs))

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        new_state, reward, done, info = self.env.step(action)
        new_state = torch.FloatTensor(new_state)
        return new_state, reward, done, info


def get_env_dimensions(env):
    """
    Returns the observation space and action space dimensions of an environment.
    """
    return env.observation_space.shape[0], env.action_space.n
