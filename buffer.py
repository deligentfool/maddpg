import numpy as np
import random
from collections import deque


class replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observations, actions, reward, next_observations, done):
        self.memory.append([observations, actions, reward, next_observations, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, dones = zip(* batch)
        return observations, actions, rewards, next_observations, dones

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()