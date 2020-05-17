import gym
import numpy
from dqn_agent import DQNAgent
from atari_wrappers import WarpFrame, FrameStack
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Pong-v0')
env = FrameStack(env, 4)

num_actions = env.action_space.n
input_shape = (4, 84, 84)


config = {'capacity': 1000000, 'device': 'cpu', 'learning_rate': 0.001,
 			'eps': 1.0, 'num_actions': num_actions, 'batch_size': 32,
 			'input_shape': input_shape, 'gamma': 0.99}

agent = DQNAgent(config)

obs = env.reset()
eps = np.linspace(1.0, 0.1, 500000)

crt_num_episodes = 0
crt_ep_reward = 0.0
total_rewards = 0.0
num_episodes = 10000
render = False

for t in range(num_episodes):
	obs = np.array(env.reset())
	done = False

	while not done:
		if render:
			env.render()

		action = agent.act(obs)
		new_obs, reward, done, _ = env.step(action)
		new_obs = np.array(new_obs)
		agent.remember(obs, action, reward, new_obs, done)
		obs = new_obs

		print(obs.shape)

		for i in range(4):
			plt.imshow(obs.mean(axis=2))
			plt.show()

		crt_ep_reward += reward

	print(crt_ep_reward)
	crt_ep_reward = 0.0

	print(len(agent.memory.buffer))
	if len(agent.memory.buffer) >= 500000:
		if t >= 300:
			agent.config['eps'] = eps[t - 300]
		print('Loss:', agent.train())