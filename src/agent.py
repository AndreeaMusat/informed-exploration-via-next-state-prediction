from abc import abstractmethod
import numpy as np


class Agent(object):
	@abstractmethod
	def get_action(self, obs):
		pass

	@abstractmethod
	def remember(self, obs, action, reward, new_obs, done):
		pass

	@abstractmethod
	def train(self):
		pass


class RandomAgent(Agent):
	def __init__(self, num_actions):
		self.num_actions = num_actions

	def get_action(self, obs):
		return np.random.choice(self.num_actions)
