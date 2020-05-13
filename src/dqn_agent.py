import dqn
import agent
import memory_replay
import torch
import gym

class DQNAgent(agent.Agent):

	def __init__(self, config):
		self.config = config
		self.q_net = dqn.QNetwork(self.config)
		self.memory = memory_replay.MemoryReplay(self.config['capacity'])
		self.opt = torch.optim.Adam(self.q_net.parameters(), 
									self.config['learning_rate'])

	def act(self, state):
		if np.random.random() > self.config['eps']:
			return torch.max(self.q_net(state)).indices.item()
		else:
			return np.random.choice(self.config['num_actions'])

	def remember(self, curr_state, action, reward, next_state, done):
		self.memory.add(curr_state, action, reward, next_state, done)

	# def train(self):
	# 	batch = self.memory.sample(self.config['batch_size'])
	# 	curr_states, actions, rewards, next_states, dones = batch

	# 	curr_qs = self.q_net(curr_states)
	# 	next_qs = self.q_net(next_states)


# env = gym.make("Pong-v0")
# num_actions = env.action_space.n

# config = {'capacity': 50, 'device': 'cpu', 'learning_rate': 0.001,
# 			'eps': 0.1, 'num_actions': num_actions, 'batch_size': 5,
# 			'input_shape': }
# agent = DQNAgent(config)
