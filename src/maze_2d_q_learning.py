# Q-Learning source code adapted from: 
# https://github.com/MattChanTK/ai-gym/blob/master/maze_2d/maze_2d_q_learning.py
# to support agents with cv-based imagination.


import numpy as np
import math
import matplotlib.pyplot as plt
import os
import random
import sys

import gym
import gym_maze

from imagination import (
    RandomImagination,
    L1Imagination,
    RandomDistillation,
)

from collections import deque
from PIL import Image

os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import random as rd
np.random.seed(102)
rd.seed(102)

simulation_number = 0
statistics = {}
cell_counts = np.zeros((10, 10)).astype(int)

idx_to_direction = {0: 'N', 1: 'S', 2: 'E', 3: 'W'}


def mark_position(game_vis, crt_pos, fig_name=None):
    crt_pos = (crt_pos[0], crt_pos[1])

    x = int((crt_pos[1] + 0.5) * 64)
    y = int((crt_pos[0] + 0.5) * 64)

    global cell_counts
    print('Cell counts: {}'.format(cell_counts[crt_pos[0], crt_pos[1]]))

    l = 27 - 8 * cell_counts[crt_pos[0], crt_pos[1]]
    l = max(l, 5)

    print(l)

    game_vis[x-l:x+l, y-l:y+l, 0:3] = 0.2
    game_vis[x-l:x+l, y-l:y+l, 3] = 0.15

    if fig_name is not None:
        plt.imshow(game_vis)
        plt.savefig(fig_name)

    cell_counts[crt_pos[0], crt_pos[1]] += 1

    return game_vis


def simulate(reset=False):
    global simulation_number
    simulation_number += 1

    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99

    num_streaks = 0
    learning_rates, idx = None, None
    
    # Render the maze
    env.render()
    
    for episode in range(NUM_EPISODES):
        if reset:
            imagination_module = RandomDistillation(num_actions=4, 
                                       memory=[], 
                                       capacity=8, 
                                       model_type='unet_concat',
                                       lr=0.3,
                                       model_path=model_path,)
            

            '''imagination_module = L1Imagination(num_actions=4, 
                                               memory=[],
                                               capacity=8,
                                               model_type='unet_concat',
                                               model_path=model_path)
               imagination_module = RandomImagination(num_actions=4)
            '''

        # Reset the environment
        obv = env.reset()
        crt_frame = env.render(mode='rgb_array')

        # Idea: visualize the exploration results by plotting the history
        # of agent positions transparently on the game image.
        if crt_frame.max() > 1:
            game_vis = Image.fromarray(crt_frame)
        else:
            game_vis = Image.fromarray(np.uint8(crt_frame * 255))
        game_vis.putalpha(255)
        game_vis = np.array(game_vis) / 255.

        stacked_frames = deque([crt_frame] * 4, 4)
        imagination_module.update(stacked_frames)
        
        # the initial state
        state_0 = state_to_bucket(obv)
        total_reward = 0

        for t in range(MAX_T):
            # Select an action
            action = select_action(state_0, explore_rate,
                                   stacked_frames, imagination_module)

           
            print('Step {}, action {}'.format(t, action))
 
            # execute the action
            obv, reward, done, _ = env.step(action)

            # Visualize the route of the agent.
            crt_cell = state_to_bucket(obv)
            print(crt_cell)

            vis_name = '../vis/sim_' + str(simulation_number) + 'ep_'
            vis_name += str(episode) + 'step_' + str(t) + '.png'
            game_vis = mark_position(game_vis, crt_cell, vis_name)

            stacked_frames.append(env.render(mode='rgb_array'))
            imagination_module.update(stacked_frames)
            
            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward

            # Update the Q based on the result
            best_q = np.amax(q_table[state])

            update = discount_factor * best_q - q_table[state_0 + (action,)]
            update += reward
            q_table[state_0 + (action,)] += update

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if DEBUG_MODE == 2:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")

            elif DEBUG_MODE == 1:
                if done or t >= MAX_T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("")

            # Render the maze
            if RENDER_MAZE:
                env.render()

            if env.is_game_over():
                sys.exit()

            if done:
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

                if episode in statistics:
                    statistics[episode].append((t, total_reward))
                else:
                    statistics[episode] = [(t, total_reward)]
   
                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)


def select_action(state, explore_rate, stacked_frames, imagination_module):
    # Select the action indicated by the imagination module.
    # This is a random action if we use eps-greedy 'imagination'.
    if random.random() < explore_rate:
        action = imagination_module.get_action(stacked_frames)
        print('Action {}'.format(idx_to_direction[action]))
    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[state]))
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == "__main__":

    # Initialize the "maze" environment
    env = gym.make("maze-random-10x10-plus-v0")

    '''
    Defining the environment related constants
    '''
    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    '''
    Learning related constants
    '''
    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

    '''
    Defining the simulation related constants
    '''
    NUM_EPISODES = 10
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    STREAK_TO_END = 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 0
    RENDER_MAZE = False
    ENABLE_RECORDING = False

    '''
    Creating a Q-Table for each state-action pair
    '''
    # q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    '''
    Begin simulation
    '''
    recording_folder = "/tmp/maze_q_learning"
    model_path = '../models/model_ConcatConditionalUNet_lr_0.001_gdl_1_ep_1_batch_60_loss_0.14686384797096252'

    if ENABLE_RECORDING:
        env.monitor.start(recording_folder, force=True)

    for ep in range(10):
        q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
        simulate(reset=True)

    for ep in range(10):
        total_steps = 0
        total_reward = 0.0
        for steps, reward in statistics[ep]:
            total_steps += steps
            total_reward += reward
        total_steps /= 10.0
        total_reward /= 10.0
        print('Ep {} avg_steps {} avg_reward {}'.format(ep, total_steps, total_reward))
        
        
    if ENABLE_RECORDING:
        env.monitor.close()
