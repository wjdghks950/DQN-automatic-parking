import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple

transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
state = namedtuple('state', ('state_tuple', 'state_img'))

class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        return out

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, time_step):
        sample_t = random.sample(list(enumerate(self.memory)), batch_size) # Retrieve (idx, item) tuple from memory
        for idx, sample in enumerate(sample_t):
            for j in range(1, time_step+1):
                sample_t[0][1] = torch.cat(sample_t[0][1], self.memory[idx][0][1]), dim=0)
        return sample_t
