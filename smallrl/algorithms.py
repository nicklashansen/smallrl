from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from smallrl import buffers
from smallrl import distributions
from smallrl import environments
from smallrl import utils
from .agent import Agent


class REINFORCE(Agent):
    def __init__(self, env, net, optimizer, discount_factor=1, entropy_weight=0):
        super().__init__(
            env=env,
            net=net,
            optimizer=optimizer
        )
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight

    def step(self, state):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        entropy = distributions.entropy(probs)

        action = distributions.sample_logits(logits)
        new_state, reward, done, _ = self.env.step(action)
        new_state = self.process_state(new_state)

        return log_probs[action], entropy, new_state, reward, done

    def generate_rollout(self, optimize=False):
        state = self.reset()
        done = False
        rollout = []
        episode_reward = 0

        while not done:
            log_prob, entropy, state, reward, done = self.step(state)
            rollout.append((log_prob, entropy, reward))
            episode_reward += reward

        if optimize:
            loss = self.compute_loss(rollout)
            self.optimize(loss)
        
        return episode_reward

    def compute_loss(self, rollout):
        R = torch.zeros(1)
        loss = torch.zeros_like(R)

        for transition in reversed(rollout):
            log_prob, entropy, reward = transition
            R = self.discount_factor * R + reward
            loss -= log_prob * R - self.entropy_weight * entropy

        return loss


class DQN(Agent):
    def __init__(self, env, net, optimizer, memory, eps_schedule, discount_factor=1, update_frequency=20, batch_size=32):
        super().__init__(
            env=env,
            net=net,
            optimizer=optimizer
        )
        self.target_net = deepcopy(net)
        self.target_net.eval()
        self.memory = memory
        self.eps_schedule = eps_schedule
        self.discount_factor = discount_factor
        self.update_frequency = update_frequency
        self.batch_size = batch_size
        self.steps_taken = 0

    def step(self, state):
        if np.random.random() > self.eps_schedule[self.steps_taken]:
            action = self.forward(state).argmax().detach()
        else:
            obs_dims, action_dims = environments.get_env_dimensions(self.env)
            action = torch.randint(action_dims, size=(1,))[0]

        new_state, reward, done, _ = self.env.step(action)
        new_state = self.process_state(new_state)

        if done:
            bp = 0

        return new_state, action, reward, done

    def generate_rollout(self, optimize=False):
        state = self.reset()
        done = False
        episode_reward = 0

        while not done:
            new_state, action, reward, done = self.step(state)
            self.memory += state, action, reward, new_state, done
            episode_reward += reward
            state = new_state
            
            if optimize:
                self.steps_taken += 1
                loss = self.compute_loss()
                if loss is not None:
                    self.optimize(loss)
                if self.steps_taken % self.update_frequency == 0:
                    self.target_net.load_state_dict(self.net.state_dict())
                    self.target_net.eval()
        
        return episode_reward

    def compute_loss(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, new_states, dones = utils.transitions_to_sars(transitions)
        terminal_mask = (1-dones).byte().float()
        
        Q = self.net(states).gather(1, actions)
        Q_next = (self.target_net(new_states).max(1)[0].view(-1, 1) * terminal_mask).detach() * self.discount_factor + rewards

        return F.mse_loss(Q, Q_next)


class A2C(Agent):
    def __init__(self, env, net, optimizer, discount_factor=1, entropy_weight=0, value_weight=0.5, update_frequency=20):
        super().__init__(
            env=env,
            net=net,
            optimizer=optimizer
        )
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.update_frequency = update_frequency

    def step(self, state):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        entropy = distributions.entropy(probs)

        action = distributions.sample_logits(logits)
        new_state, reward, done, _ = self.env.step(action)
        new_state = self.process_state(new_state)

        return log_probs[action], value, entropy, new_state, reward, done

    def generate_rollout(self, optimize=False):
        state = self.reset()
        done = False
        rollout = []
        episode_reward = 0

        while not done:
            log_prob, value, entropy, state, reward, done = self.step(state)
            rollout.append((log_prob, value, entropy, reward))
            episode_reward += reward

            if optimize and len(rollout) == self.update_frequency:
                loss = self.compute_loss(rollout)
                self.optimize(loss)
                rollout = []
            
        return episode_reward

    def compute_loss(self, rollout):
        R = torch.zeros(1)
        policy_loss = torch.zeros_like(R)
        value_loss = torch.zeros_like(R)
        next_value = torch.zeros_like(R)
        gae = torch.zeros_like(R)

        for transition in reversed(rollout):
            log_prob, value, entropy, reward = transition
            R = self.discount_factor * R + reward

            advantage = R - value
            value_loss += 0.5 * advantage.pow(2)

            td_error = reward + self.discount_factor * next_value.data - value.data
            gae = gae * self.discount_factor + td_error
            next_value = value.data

            policy_loss -= log_prob * gae - self.entropy_weight * entropy

        return policy_loss + self.value_weight * value_loss


class ACER(A2C):
    def __init__(self, env, net, optimizer, memory, discount_factor=1, entropy_weight=0, value_weight=0.5, update_frequency=20, batch_size=4):
        super().__init__(
            env=env,
            net=net,
            optimizer=optimizer,
            discount_factor=discount_factor,
            entropy_weight=entropy_weight,
            value_weight=value_weight,
            update_frequency=update_frequency
        )
        self.memory = memory
        self.batch_size = batch_size

    def generate_rollout(self, optimize=False):
        state = self.reset()
        done = False
        rollout = []
        episode_reward = 0

        while not done:
            log_probs, value, entropy, state, reward, done = self.step(state)
            rollout.append((log_probs, value, entropy, state, reward))
            episode_reward += reward

            if optimize and len(rollout) == self.update_frequency:
                loss = self.compute_loss(rollout, on_policy=True)
                self.optimize(loss)

                if len(self.memory) >= self.batch_size:
                    replay_rollouts = self.memory.sample(self.batch_size)
                    for replay_rollout in replay_rollouts:
                        loss = self.compute_loss(replay_rollout, on_policy=False)
                        self.optimize(loss)

                self.memory += rollout
                rollout = []
            
        return episode_reward

    def compute_loss(self, rollout, on_policy=False):
        R = torch.zeros(1)
        policy_loss = torch.zeros_like(R)
        value_loss = torch.zeros_like(R)
        next_value = torch.zeros_like(R)
        gae = torch.zeros_like(R)

        for transition in reversed(rollout):
            log_prob, value, entropy, state, reward = transition
            
            if not on_policy:
                logits, value = self.forward(state)
                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs)
                entropy = distributions.entropy(probs)
                action = distributions.sample_logits(logits)
                log_prob = log_probs[action]
            
            R = self.discount_factor * R + reward
            advantage = R - value
            value_loss += 0.5 * advantage.pow(2)

            td_error = reward + self.discount_factor * next_value.data - value.data
            gae = gae * self.discount_factor + td_error
            next_value = value.data

            policy_loss -= log_prob * gae - self.entropy_weight * entropy

        return policy_loss + self.value_weight * value_loss
