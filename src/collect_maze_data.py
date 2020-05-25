"""
Data collection script. The data will be stored in the following way:

- root_dir
    - env_0
        - ep_0
            - 0_*.png
            - 1_*.png
            - ...
        - ep_1
            - 0_*.png
            - 1_*.png
            ...
        - ...
    - env_1
        - ep_1
            - 0_*.png
            - 1_*.png
            - ...
        - ...
    - ...
"""

import matplotlib
import os
import shutil

import gym
import gym_maze

from matplotlib import image
from skimage import color


num_envs = 1000
num_episodes = 50
max_frames = 2500

# Warning: this code will delete all existent data in
# data_path and will start collecting the data again.
# Put a dummy folder name to avoid deleting everything.
data_path = os.path.join('.', 'TODO-CHANGE-THIS-NAME')
if not os.path.exists(data_path):
    os.mkdir(data_path)

for env_no in range(num_envs):
    env = gym.make("maze-random-10x10-plus-v0")
    env_data_dir = os.path.join(data_path, 'env_' + str(env_no))
    if os.path.exists(env_data_dir):
        shutil.rmtree(env_data_dir)
    os.mkdir(env_data_dir)        

    for ep_no in range(num_episodes):
        ep_dir = os.path.join(env_data_dir, 'ep_' + str(ep_no))
        if os.path.exists(ep_dir):
            shutil.rmtree(ep_dir)
        os.mkdir(ep_dir)                

        obs = env.reset()
        for frame_no in range(max_frames):
            pixels = env.render(mode='rgb_array')
            pixels = color.rgb2gray(pixels)

            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)

            # Save the name of the current frame in the format
            # XXX_Y.png, where XXX is the current frame number and
            # Y is the current action the agent took.
            f_name = str(frame_no) + '_' + str(action)
            f_path = os.path.join(ep_dir, f_name + '.png')
            matplotlib.image.imsave(f_path, pixels)

            # Also save the last frame of the game.
            if done:
                frame_no += 1
                pixels = env.render(mode='rgb_array')
                pixels = color.rgb2gray(pixels)
                action = env.action_space.sample()
                f_name = str(frame_no) + '_' + str(action)
                f_path = os.path.join(ep_dir, f_name + '.png')
                matplotlib.image.imsave(f_path, pixels)

                print('Finished episode %d after %d steps' % (
                    ep_no,
                    frame_no
                ))
                
                break

    del env
