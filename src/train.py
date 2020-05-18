import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import os
import torch


from atari_wrappers import wrap_deepmind
from dqn_agent import DQNAgent
from utils import *


root_path = mount_drive()

env = gym.make('Pong-v0')
env = wrap_deepmind(env, frame_stack=True, scale=True)
num_actions = env.action_space.n
config = get_default_config(num_actions)
agent = DQNAgent(config)

crt_num_episodes = 0
crt_ep_reward = 0.0
total_rewards = 0.0
render = False


loss_hist = []
avg_rewards = []

i = 0
while agent.config['episodes_left']:
  obs = np.array(env.reset())
  
  done = False
  while not done:
    if render:
      env.render()

    action = agent.act(obs)
    new_obs, reward, done, _ = env.step(action)
    new_obs = np.array(new_obs)
    agent.remember(obs, action, reward, new_obs, done)
    
    # fname = get_img_name(agent.config, action)
    # matplotlib.image.imsave(fname, obs[:, :, 3])
    
    obs = new_obs
    crt_ep_reward += reward

    if len(agent.memory.buffer) >= 5000:
      loss = agent.train()
      if i % 100 == 0:
        print('loss:', loss)
    
  print('eps:', agent.config['eps'])
  print('rew:', crt_ep_reward)

  loss_hist.append(agent.train())
  avg_rewards.append(crt_ep_reward)

  crt_ep_reward = 0.0

  new_eps = agent.config['eps'] * 0.99
  agent.config['eps'] = max(new_eps, 0.05)
  agent.config['episodes_left'] -= 1

  if agent.config['episodes_left'] % 2 == 0:
    create_checkpoint(agent, loss_hist, avg_rewards, root_path)