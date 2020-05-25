import gym
import os
import sys
import torch

from dqn_agent import DQNAgent


def get_default_config():
  input_shape = (4, 84, 84)
  config = {
    'batch_size': 32,
    'capacity': 1000, 
    'device': 'cpu',
    'env': 'Pong-v0',
    'episodes_left' : 10000,
    'eps': 0.9,
    'eps_decay': 0.99,
    'eps_min': 0.05,
    'gamma': 0.99,
    'img_idx': 0,
    'input_shape': input_shape, 
    'learning_rate': 0.001,
    'total_episodes': 10000,
    'unique_name': 'agent0001',
  }
  env = gym.make('Pong-v0')

  config['num_actions'] = env.action_space.n
  
  return config


def mount_drive():
  from google.colab import drive
  drive.mount('/content/drive')
  root_path =  '/content/drive/My Drive/'
  sys.path.insert(1, root_path)
  sys.path.insert(1, os.path.join(root_path, 'src'))
  return root_path


def get_img_name(my_config, action, root_path='/content/drive/My Drive/'):
  fname = 'frame_' + str(my_config['img_idx']) 
  fname += '_act' + str(action) + '.png'
  fname = os.path.join(root_path, 'data', my_config['unique_name'], fname)
  my_config['img_idx'] += 1
  return fname


def create_checkpoint(agent, 
                      loss_hist, 
                      avg_rewards, 
                      root_path='/content/drive/My Drive/'):
  checkpoint_dict = {
    'q_net': agent.q_net.state_dict(),
    'config': agent.config,
    'buffer': agent.memory.buffer,
    'loss_hist': loss_hist,
    'avg_rewards': avg_rewards,
  }

  crt_episode = agent.config['total_episodes']
  crt_episode -= agent.config['episodes_left']
  checkpoint_fname = os.path.join(root_path, 'my_checkpoints')
  checkpoint_fname = os.path.join(checkpoint_fname, str(crt_episode) + '.pth')

  torch.save(checkpoint_dict, checkpoint_fname)


def load_from_checkpoint(checkpoint_fname):
  checkpoint_dict = torch.load(checkpoint_fname)

  agent = DQNAgent(checkpoint_dict['config'])
  agent.q_net.load_state_dict(checkpoint_dict['q_net'])
  agent.memory.buffer = checkpoint_dict['buffer']
  loss_hist = checkpoint_dict['loss_hist']
  avg_rewards = checkpoint_dict['avg_rewards']

  return agent, loss_hist, avg_rewards
