import random
import torch
import numpy as np

class MemoryReplay(object):

  def __init__(self, config):
    self.device = config['device']
    self.capacity = config['capacity']
    self.buffer = []

  def add(self, curr_state, action, reward, next_state, done):
    if len(self.buffer) == self.capacity:
      self.buffer.pop(0)

    curr_state = np.array(curr_state).transpose(2, 0, 1) / 255.0
    next_state = np.array(next_state).transpose(2, 0, 1) / 255.0	

    self.buffer.append((curr_state, action, reward,
              next_state, done))

  def sample(self, batch_size):
    batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
    curr_states, actions, rewards, next_states, dones = zip(*batch)

    curr_states = torch.Tensor(curr_states).double().to(self.device)
    actions = torch.Tensor(actions).long().to(self.device)
    rewards = torch.Tensor(rewards).double().to(self.device)
    next_states = torch.Tensor(next_states).double().to(self.device)
    dones = torch.Tensor(dones).long().to(self.device)

    return curr_states, actions, rewards, next_states, dones
