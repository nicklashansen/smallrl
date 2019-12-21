import torch
import numpy as np
import gym

import algorithms
import buffers
import networks
import environments
import utils
import schedules


def train_agent(agent, criterion, verbose=True):
    """
    Runs a training loop for a given agent.
    """
    if verbose:
        print(f'Training {agent.algorithm} agent on {agent.env.unwrapped.spec.id}...')

    running_reward = utils.RunningMean(n=criterion['n'])
    target_reward = criterion['target_reward']

    for episode in range(1, criterion['max_episodes']+1):
        reward = agent.generate_rollout(optimize=True)
        running_reward += reward
        
        if verbose and episode % criterion['n'] == 0:
            print(f'Episode {episode}, mean reward: {running_reward}')

        if running_reward.mean() >= target_reward:
            print(f'Reached target reward of {target_reward} in {episode} episodes!')
            break

    print('Training has been terminated.')


def init_reinforce_agent(env):
    obs_dims, action_dims = environments.get_env_dimensions(env)
    net = networks.MLP(obs_dims, 64, action_dims)

    return algorithms.REINFORCE(
        env=env,
        net=net,
        optimizer=torch.optim.RMSprop(net.parameters(), lr=5e-3),
        discount_factor=0.95,
        entropy_weight=0
    )


def init_dqn_agent(env):
    obs_dims, action_dims = environments.get_env_dimensions(env)
    net = networks.MLP(obs_dims, 256, action_dims) 

    return algorithms.DQN(
        env=env,
        net=net,
        optimizer=torch.optim.RMSprop(net.parameters(), lr=5e-4),
        memory=buffers.ReplayMemory(capacity=50000),
        eps_schedule=schedules.LinearEpsilonSchedule(eps_start=0.3, eps_end=0.02, num_steps=100000),
        discount_factor=0.98
    )


def init_a2c_agent(env):
    obs_dims, action_dims = environments.get_env_dimensions(env)
    net = networks.TwoHeadedMLP(obs_dims, 128, action_dims)

    return algorithms.A2C(
        env=env,
        net=net,
        optimizer=torch.optim.RMSprop(net.parameters(), lr=5e-4),
        discount_factor=0.98,
        entropy_weight=0,
        value_weight=0.5,
        update_frequency=20
    )


def init_acer_agent(env):
    obs_dims, action_dims = environments.get_env_dimensions(env)
    net = networks.TwoHeadedMLP(obs_dims, 128, action_dims)

    return algorithms.ACER(
        env=env,
        net=net,
        optimizer=torch.optim.RMSprop(net.parameters(), lr=5e-4),
        memory=buffers.ReplayMemory(capacity=1000),
        discount_factor=0.98,
        entropy_weight=0,
        value_weight=0.5,
        update_frequency=10,
        batch_size=2
    )


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = environments.TorchEnv(env)
    criterion = {'n': 100, 'target_reward': 195, 'max_episodes': 2000}

    # REINFORCE
    train_agent(agent=init_reinforce_agent(env), criterion=criterion, verbose=True)

    # DQN
    train_agent(agent=init_dqn_agent(env), criterion=criterion, verbose=True)

    # A2C
    train_agent(agent=init_a2c_agent(env), criterion=criterion, verbose=True)

    # ACER
    train_agent(agent=init_acer_agent(env), criterion=criterion, verbose=True)
