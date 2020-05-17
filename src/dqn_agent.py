import dqn
import agent
import memory_replay
import torch
import gym
import matplotlib.pyplot as plt
import numpy as np
from atari_wrappers import WarpFrame, FrameStack


class DQNAgent(agent.Agent):

	def __init__(self, config):
		self.config = config
		self.q_net = dqn.QNetwork(self.config).double()
		self.memory = memory_replay.MemoryReplay(self.config)
		self.opt = torch.optim.Adam(self.q_net.parameters(), 
									self.config['learning_rate'])

	def act(self, state):
		state = np.array(state).transpose(2, 0, 1)[None, :] / 255.0
		state = torch.Tensor(state).double()

		if np.random.random() > self.config['eps']:
			return torch.max(self.q_net(state), 1).indices[0].item()
		else:
			return np.random.choice(self.config['num_actions'])

	def remember(self, curr_state, action, reward, next_state, done):
		self.memory.add(curr_state, action, reward, next_state, done)

	def train(self):
		batch = self.memory.sample(self.config['batch_size'])
		curr_states, actions, rewards, next_states, dones = batch

		curr_qs = self.q_net(curr_states)
		next_qs = self.q_net(next_states)

		curr_qs = curr_qs.gather(1, actions.unsqueeze(1)).squeeze(1)
		next_qs = next_qs.max(1)[0]

		target_qs = rewards + self.config['gamma'] * next_qs * (1 - dones)
		loss = (curr_qs - target_qs.detach()).pow(2).mean()

		self.opt.zero_grad()
		loss.backward()

		self.opt.step()
		return loss.item()

