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

# root_path = mount_drive()
root_path = '.'

restore_from_checkpoint = False
if not restore_from_checkpoint:
  agent = DQNAgent(get_default_config())
  loss_hist = []
  avg_rewards = []
else:
  checkpoint_fname = os.path.join(root_path, 'my_checkpoints', '2.pth')  # TODO: change checkpoint if needed!!!!
  checkpoint = load_from_checkpoint(checkpoint_fname)
  agent, loss_hist, avg_rewards = checkpoint
  
  agent_code = int(agent.config['unique_name'][-4:]) + 1
  agent_code_str = str(agent_code)
  agent_code_str = "0" * (4 - len(agent_code_str)) + agent_code_str
  agent.config['unique_name'] = 'agent' + str(agent_code)

# data_dir = os.path.join(root_path, 'data', agent.config['unique_name'])
# if not os.path.exists(data_dir):
#   os.mkdir(data_dir)

# checkpoint_dir = os.path.join(root_path, 'my_checkpoints')
# if not os.path.exists(checkpoint_dir):
  # os.mkdir(checkpoint_dir)

env = wrap_deepmind(gym.make('Pong-v0'), 
                    frame_stack=True, 
                    scale=True)

crt_num_episodes = 0
crt_ep_reward = 0.0
total_rewards = 0.0
render = False

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

    i += 1
    if len(agent.memory.buffer) >= 500:
      loss = agent.train()
      if i % 100 == 0:
        print('loss:', loss)
    
  print('eps:', agent.config['eps'])
  print('rew:', crt_ep_reward)

  loss_hist.append(agent.train())
  avg_rewards.append(crt_ep_reward)

  crt_ep_reward = 0.0

  agent.mark_episode()

  if agent.config['episodes_left'] % 10 == 0:
    print('Wrote checkpoint!')
    create_checkpoint(agent, loss_hist, avg_rewards, root_path)