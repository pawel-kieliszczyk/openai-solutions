import collections
import random

import numpy as np


class ReplayMemory(object):
    def __init__(self, size_limit):
        self.data = collections.deque(maxlen=size_limit)

    def add(self, state, action, reward, next_state, done):
        self.data.append((state, action, reward, next_state, done))

    def get_samples(self, batch_size):
        if len(self.data) < batch_size:
            batch_size = len(self.data)

        samples = random.sample(self.data, batch_size)
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
        return states_batch, action_batch, reward_batch, next_states_batch, done_batch
