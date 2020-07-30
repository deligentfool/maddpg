import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gym
import make_env
import math
from buffer import replay_buffer
from net import policy_net, value_net
import os
import time
import copy


class maddpg(object):
    def __init__(self, env_id, episode, learning_rate, gamma, capacity, batch_size, value_iter, policy_iter, rho, episode_len, render, train_freq, save_freq, entropy_weight):
        self.env_id = env_id
        self.env = make_env.make_env(self.env_id)
        self.episode = episode
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.capacity = capacity
        self.batch_size = batch_size
        self.value_iter = value_iter
        self.policy_iter = policy_iter
        self.rho = rho
        self.render = render
        self.episode_len = episode_len
        self.train_freq = train_freq
        self.save_freq = save_freq
        self.entropy_weight = entropy_weight

        self.observation_dims = self.env.observation_space
        self.action_dims = self.env.action_space
        self.observation_total_dims = sum([self.env.observation_space[i].shape[0] for i in range(self.env.n)])
        self.action_total_dims = sum([self.env.action_space[i].n for i in range(self.env.n)])
        self.policy_nets = [policy_net(self.observation_dims[i].shape[0], self.action_dims[i].n) for i in range(self.env.n)]
        self.target_policy_nets = [policy_net(self.observation_dims[i].shape[0], self.action_dims[i].n) for i in range(self.env.n)]
        self.value_nets = [value_net(self.observation_total_dims, self.action_total_dims, 1) for i in range(self.env.n)]
        self.target_value_nets = [value_net(self.observation_total_dims, self.action_total_dims, 1) for i in range(self.env.n)]
        self.policy_optimizers = [torch.optim.Adam(policy_net.parameters(), lr=self.learning_rate) for policy_net in self.policy_nets]
        self.value_optimizers = [torch.optim.Adam(value_net.parameters(), lr=self.learning_rate) for value_net in self.value_nets]
        [target_policy_net.load_state_dict(policy_net.state_dict()) for target_policy_net, policy_net in zip(self.target_policy_nets, self.policy_nets)]
        [target_value_net.load_state_dict(value_net.state_dict()) for target_value_net, value_net in zip(self.target_value_nets, self.value_nets)]
        self.buffer = replay_buffer(self.capacity)
        self.count = 0
        self.train_count = 0

    def soft_update(self):
        for i in range(self.env.n):
            for param, target_param in zip(self.value_nets[i].parameters(), self.target_value_nets[i].parameters()):
                target_param.detach().copy_(self.rho * target_param.detach() + (1. - self.rho) * param.detach())
            for param, target_param in zip(self.policy_nets[i].parameters(), self.target_policy_nets[i].parameters()):
                target_param.detach().copy_(self.rho * target_param.detach() + (1. - self.rho) * param.detach())

    def train(self):
        observations, actions, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

        observations_stack = np.vstack([np.hstack(observations[i]) for i in range(len(observations))])
        total_observations = torch.FloatTensor(observations_stack).view(len(observations), -1)
        indiv_observations = []
        for i in range(self.env.n):
            indiv_observations.append(torch.FloatTensor(np.vstack([observations[b][i] for b in range(self.batch_size)])))
        actions_stack = np.vstack([np.hstack(actions[i]) for i in range(len(actions))])
        indiv_actions = []
        for i in range(self.env.n):
            indiv_actions.append(torch.FloatTensor(np.vstack([actions[b][i] for b in range(self.batch_size)])))
        total_actions = torch.cat(indiv_actions, dim=1)
        rewards = torch.FloatTensor(rewards)
        indiv_rewards = []
        for i in range(self.env.n):
            indiv_rewards.append(rewards[:, i])
        next_observations_stack = np.vstack([np.hstack(next_observations[i]) for i in range(len(next_observations))])
        total_next_observations = torch.FloatTensor(next_observations_stack).view(len(next_observations), -1)
        indiv_next_observations = []
        for i in range(self.env.n):
            indiv_next_observations.append(torch.FloatTensor(np.vstack([next_observations[b][i] for b in range(self.batch_size)])))
        dones = torch.FloatTensor(dones)
        indiv_dones = []
        for i in range(self.env.n):
            indiv_dones.append(dones[:, i])

        for _ in range(self.value_iter):
            target_next_actions = torch.cat([self.target_policy_nets[i].forward(indiv_next_observations[i])[0] for i in range(self.env.n)], dim=1)
            for i in range(self.env.n):
                target_next_value = self.target_value_nets[i].forward(total_next_observations, target_next_actions)
                q_target = indiv_rewards[i].unsqueeze(1) + self.gamma * (1 - indiv_dones[i].unsqueeze(1)) * target_next_value
                q_target = q_target.detach()
                q = self.value_nets[i].forward(total_observations, total_actions)
                value_loss = (q - q_target).pow(2).mean()

                self.value_optimizers[i].zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 0.5)
                self.value_optimizers[i].step()

        for _ in range(self.policy_iter):
            for i in range(self.env.n):
                prob, entropy = self.policy_nets[i].forward(indiv_observations[i])
                #action_idx = torch.LongTensor(actions)[:, i].unsqueeze(1)
                #action_prob = prob.gather(dim=1, index=action_idx)
                #policy_loss = -self.value_nets[i].forward(total_observations, total_actions).detach() * action_prob.log() - self.entropy_weight * entropy

                new_action = copy.deepcopy(indiv_actions)
                new_action[i] = prob
                new_actions = torch.cat(new_action, dim=1)
                policy_loss = - self.value_nets[i].forward(total_observations, new_actions) - self.entropy_weight * entropy

                policy_loss = policy_loss.mean()

                self.policy_optimizers[i].zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_nets[i].parameters(), 0.5)
                self.policy_optimizers[i].step()

        self.soft_update()
        #self.buffer.clear()

    def run(self):
        weight_reward = [None for i in range(self.env.n)]
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = [0 for i in range(self.env.n)]
            if self.render:
                self.env.render()
            while True:
                actions = []
                for n in range(self.env.n):
                    action = self.policy_nets[n].act(torch.FloatTensor(np.expand_dims(obs[n], 0)))
                    actions.append(action)
                next_obs, reward, done, info = self.env.step(actions)
                if self.render:
                    self.env.render()
                self.buffer.store(obs, actions, reward, next_obs, done)
                self.count += 1
                total_reward = [total_reward[i] + reward[i] for i in range(self.env.n)]
                obs = next_obs

                if self.count % self.train_freq == 0:
                    self.train_count += 1
                    self.train()
                if self.count % self.save_freq == 0:
                    [torch.save(self.policy_nets[i], './models/{}/policy_model{}.pkl'.format(self.env_id, i)) for i in range(self.env.n)]
                if any(done) or self.count % self.episode_len == 0:
                    for n in range(self.env.n):
                        if weight_reward[n] is not None:
                            weight_reward[n] = weight_reward[n] * 0.99 + total_reward[n] * 0.01
                        else:
                            weight_reward[n] = total_reward[n]
                    print(('episode: {}\treward: '+'{:.2f}\t' * self.env.n).format(i + 1, *weight_reward))
                    break

    def eval(self):
        self.count = 0
        for i in range(self.env.n):
            self.policy_nets[i] = torch.load('./models/{}/policy_model{}.pkl'.format(self.env_id, i))
        while True:
            obs = self.env.reset()
            total_reward = [0 for i in range(self.env.n)]
            if self.render:
                self.env.render()
            while True:
                actions = [np.zeros(self.env.action_space[i].n) for i in range(self.env.n)]
                actions_indice = []
                for n in range(self.env.n):
                    action_idx = self.policy_nets[n].act(torch.FloatTensor(np.expand_dims(obs[n], 0)))
                    actions[n][action_idx] = 1
                    actions_indice.append(action_idx)
                next_obs, reward, done, info = self.env.step(actions)
                if self.render:
                    self.env.render()
                total_reward = [total_reward[i] + reward[i] for i in range(self.env.n)]
                obs = next_obs
                self.count += 1
                if any(done) or self.count % self.episode_len == 0:
                    print('episode: {}\treward: {}'.format(i + 1, total_reward))
                    break


if __name__ == '__main__':
    env_id = 'simple_tag'
    os.makedirs('./models/{}'.format(env_id), exist_ok=True)
    test = maddpg(
        env_id=env_id,
        episode=150000,
        learning_rate=5e-3,
        gamma=0.97,
        capacity=100000,
        batch_size=128,
        value_iter=1,
        policy_iter=1,
        rho=0.995,
        render=False,
        episode_len=50,
        train_freq=128,
        save_freq=100,
        entropy_weight=0.001
    )
    test.run()
    #test.eval()