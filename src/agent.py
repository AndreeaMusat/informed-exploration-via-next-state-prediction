from abc import abstractmethod
import numpy as np


class Agent(object):
	@abstractmethod
	def get_action(self, state):
		pass

	@abstractmethod
	def remember(self, curr_state, action, reward, next_state, done):
		pass

	@abstractmethod
	def train(self):
		pass


class RandomAgent(Agent):
	def __init__(self, num_actions):
		self.num_actions = num_actions

	def act(self, obs):
		return np.random.choice(self.num_actions)
