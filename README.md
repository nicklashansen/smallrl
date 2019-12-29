# smallrl

smallrl is a personal repository for quick prototyping of RL algorithms and applications in PyTorch. Contains extensible agents, networks, environment wrappers, buffers and decay schedules as well as implementations of REINFORCE, DQN, A2C and ACER.

Project is currently in its early stages. Work in progress!

_____

## How to use

It's easy to get started with smallrl. Here's an example of how to train an agent on CartPole with policy gradient:

```
import torch
import gym
from smallrl import environments, demos

if __name__ == '__main__':
    env = environments.TorchEnv(gym.make('CartPole-v0'))
    obs_dims, action_dims = environments.get_env_dimensions(env)
    net = networks.MLP(obs_dims, 64, action_dims)
    agent = algorithms.REINFORCE(
        env=env,
        net=net,
        optimizer=torch.optim.RMSprop(net.parameters(), lr=5e-3),
        discount_factor=0.95,
        entropy_weight=0
    )
    demos.train_agent(
        agent=agent,
        criterion={'n': 100, 'target_reward': 195, 'max_episodes': 2000},
        verbose=True
    )
```

That's it! It can be run on a CPU and should converge within a few hundred episodes.
