import random
import torch
import numpy as np

class MemoryReplay(object):

	def __init__(self, config):
		self.config = config
		self.capacity = self.config['capacity']
		self.buffer = []

	def add(self, curr_state, action, reward, next_state, done):
		if len(self.buffer) == self.capacity:
			self.buffer.pop(0)

		self.buffer.append((curr_state[None, :], action, reward,
							next_state[None, :], done))

	def sample(self, batch_size):
		batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
		curr_states, actions, rewards, next_states, dones = zip(*batch)

		curr_states = torch.Tensor(curr_states).to(self.config['device']).double()
		actions = torch.Tensor(actions).to(self.config['device']).double()
		rewards = torch.Tensor(rewards).to(self.config['device']).double()
		next_states = torch.Tensor(next_states).to(self.config['device']).double()
		dones = torch.Tensor(dones).to(self.config['device']).double()

		return curr_states, actions, rewards, next_states, dones


"""
replay = MemoryReplay({'capacity': 10, 'device': 'cpu'})

for i in range(10):
	replay.add(np.array([1]), 2, 3, np.array([4]), True)

for i in range(5):
	print(replay.sample(5))
"""