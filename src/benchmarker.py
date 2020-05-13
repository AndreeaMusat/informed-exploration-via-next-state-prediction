import gym 


class GameBenchmarker(object):
	def __init__(self, env, num_episodes, render=False):
		self.env = env
		self.render = render
		self.num_episodes = num_episodes

	def benchmark_agent(self, agent):
		crt_num_episodes = 0
		crt_ep_reward = 0
		total_rewards = 0

		obs = self.env.reset()
		while True:
			if self.render:
				env.render()

			action = agent.get_action(obs)
			new_obs, reward, done, lives = self.env.step(action)
			agent.remember(obs, action, reward, new_obs, done, lives)
			obs = new_obs
			crt_ep_reward += reward

			if done:
				total_rewards += crt_ep_reward
				crt_ep_reward = 0
				crt_num_episodes += 1

				if crt_num_episodes == self.num_episodes:
					break

		return total_rewards / crt_num_episodes
