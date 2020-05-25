import gym 


class GameBenchmarker(object):
  def __init__(self, env, num_episodes, num_frames=4, render=False):
    self.env = env
    self.render = render
    self.num_episodes = num_episodes
    self.num_frames = num_frames

  def benchmark_agent(self, agent):
    crt_num_episodes = 0
    crt_ep_reward = 0.0
    total_rewards = 0.0

    for t in range(self.num_episodes):
      obs = self.env.reset()
      frames = [obs for _ in range(self.num_frames)]

      done = False
      while not done:
        if self.render:
          self.env.render()

        action = agent.get_action(obs)
        new_obs, reward, done, _ = self.env.step(action)
        agent.remember(obs, action, reward, new_obs, done)
        obs = new_obs
        crt_ep_reward += reward
        frames.pop(0)
        frames.append(obs)

      total_rewards += crt_ep_reward
      crt_ep_reward = 0.0

    return total_rewards / self.num_episodes
