import torch
import numpy as np
import gym

from smallrl import algorithms
from smallrl import buffers
from smallrl import networks
from smallrl import environments
from smallrl import utils
from smallrl import schedules
from smallrl import demos


if __name__ == '__main__':
    print('Running smallrl example usage.\nFor more demos, visit https://github.com/nicklashansen/smallrl \n')

    env = gym.make('CartPole-v0')
    env = environments.TorchEnv(env)
    criterion = {'n': 100, 'target_reward': 195, 'max_episodes': 2000}

    # REINFORCE
    demos.train_agent(agent=demos.init_reinforce_agent(env), criterion=criterion, verbose=True)