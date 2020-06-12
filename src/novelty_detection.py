import collections
import cv2
import torch

import numpy as np
import os
import operator

from abc import abstractmethod
from PIL import Image
from skimage import color

from unet_model_cp_new import UNetConcat

models = {'unet_concat' : UNetConcat(n_channels=4, n_classes=1) }


class Imagination(object):
    """Base class for imagination modules."""

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def update(self, stacked_frames):
        pass

    
class RandomImagination(Imagination):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def get_action(self, frame):
        return np.random.choice(self.num_actions)



class L1Imagination(Imagination):
    """Imagination module for a Q-Learning based agent."""

    def __init__(self, num_actions, memory, capacity, model_type, model_path):
        super(L1Imagination, self).__init__()

        self.num_actions = num_actions
        # memory` is a list of previously seen frames.
        self._memory = collections.deque(memory, capacity)

        self.device = 'cpu'
        
        self.model = models[model_type]
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval().to(self.device)
        

    def preprocess_frame(self, frame):        
        frame = np.array(frame).astype(float) / (255 / 2)
        frame -= 1
            
        if len(frame.shape) == 3:
            frame = color.rgb2gray(frame)
            frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        return frame
     
    def get_batch(self, stacked_frames):
        for i, frame in enumerate(stacked_frames):
            stacked_frames[i] = self.preprocess_frame(frame)
            
        stacked_frames = np.array(stacked_frames)        
        stacked_frames = torch.from_numpy(stacked_frames)

        batch_frame = torch.zeros(4, *stacked_frames.shape)
        batch_frame[..., :] = stacked_frames
        batch_frame.to(self.device)

        return batch_frame

    def l1_avg_dist(self, predicted_frames):
        dists = {}

        for i, predicted_frame in enumerate(predicted_frames):
            dist = 0.0
            for mem_frame in self._memory:
                dist += np.abs(predicted_frame - mem_frame).sum()

            dist /= len(self._memory)
            dists[i] = dist

        return dists

    def update(self, stacked_frames):
        self._memory.append(self.preprocess_frame(stacked_frames[-1]))

        
    def get_action(self, stacked_frames):
        actions = torch.Tensor(np.eye(self.num_actions))
        actions.to(self.device)
        
        batch_frames = self.get_batch(stacked_frames).to(self.device)

        predicted_frames = self.model(batch_frames, actions)
        predicted_frames = predicted_frames.detach().cpu().numpy()

        dists = self.l1_avg_dist(predicted_frames)
        best_action = max(dists.items(), key=operator.itemgetter(1))[0]

        return best_action



class RandomDistillation(Imagination):
    """Imagination module via random distillation"""
    pass
