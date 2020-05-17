import numpy as np
import gym 
from agent import RandomAgent
from benchmarker import GameBenchmarker


if __name__ == '__main__':
  env = gym.make("Pong-v0")
  num_actions = env.action_space.n
  agent = RandomAgent(num_actions)
  gb = GameBenchmarker(env, 10, render=True)
  mean_rew = gb.benchmark_agent(agent)
  print(mean_rew)
