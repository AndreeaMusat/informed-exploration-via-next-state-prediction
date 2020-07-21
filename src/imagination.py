import collections
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import operator

from abc import abstractmethod
from PIL import Image
from skimage import color

from distillation_networks import (
    RandomNet,
    EnvNet,
)
from unet_models import (
    AdditiveConditionalUNet,
    ConcatConditionalUNet,
)


models = {'unet_concat' : ConcatConditionalUNet(n_channels=4, n_classes=1, action_channels=64)}


def preprocess_frame(frame):
    frame = (frame - frame.min()) / (frame.max() - frame.min())

    frame += np.random.normal(0, 0.04, frame.shape)
    frame = (frame - 0.65491969107) / 0.2275179

    if len(frame.shape) == 3:
        frame = color.rgb2gray(frame)
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    return frame


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
    """ L1 Imagination module for a Q-Learning based agent.
    
    Keep a fixed capacity memory of the last frames seen on the
    current trajectory. When making a decision, choose the action
    which leads to the next frame prediction that maximizes the
    average L1 distance to the frames in the history.
    """
    def __init__(self, num_actions, memory, capacity, model_type, model_path):
        super(L1Imagination, self).__init__()

        self.num_actions = num_actions
        
        # memory` is a list of previously seen frames.
        self._memory = collections.deque(memory, capacity)

        self.device = 'cpu'
        
        self.model = models[model_type]
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval().to(self.device)
     
    def get_batch(self, stacked_frames):
        for i, frame in enumerate(stacked_frames):
            stacked_frames[i] = preprocess_frame(frame)
            
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
        self._memory.append(preprocess_frame(stacked_frames[-1]))
  
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
    def __init__(self, num_actions, memory,
                 capacity, model_type, model_path, train_every=1, 
                 lr=0.6, dist_type='mse', view_results=False):
        super(RandomDistillation, self).__init__()

        self.num_actions = num_actions
        self.cnt = 0

        # memory` is a list of previously seen frames.
        self._memory = collections.deque(memory, capacity)
        self.train_every = train_every
        self.train_count = 0
        self.update_count = 0
        self.device = 'cuda'
        self.view_results = view_results
        self.dist_type = dist_type

        self.model = models[model_type].cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval().to(self.device)

        self.env_model = EnvNet().to(self.device)
        self.opt = torch.optim.Adam(self.env_model.parameters(), lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma=0.999)

        self.random_net = RandomNet().to(self.device)

    def get_batch(self, stacked_frames):
        for i, frame in enumerate(stacked_frames):
            stacked_frames[i] = preprocess_frame(frame)
            
        stacked_frames = np.array(stacked_frames)        
        stacked_frames = torch.from_numpy(stacked_frames)

        batch_frame = torch.zeros(4, *stacked_frames.shape)
        batch_frame[..., :] = stacked_frames
        batch_frame.to(self.device)

        return batch_frame

    def get_criterion(self):
        if self.dist_type == 'mse':
            return lambda x, y: nn.MSELoss()(x, y)
        elif self.dist_type == 'cosine_dist':
            return lambda x, y: 1.0 - nn.CosineSimilarity()(x, y)
        elif self.dist_type == 'l1':
            return lambda x, y: nn.L1Loss()(x, y)
        else:
            raise NotImplementedError

    def train(self, batch_size, epochs=1, verbose=False):
        self.env_model.train()
       
        criterion = self.get_criterion()

        print('{}\'th time training environment model...'.format(self.train_count + 1))
        self.train_count += 1
        
        for epoch in range(epochs):
            for batch_idx in range(len(self._memory) // batch_size):
                batch_mask = np.random.choice(len(self._memory), batch_size, replace=False)
                frames = np.array(self._memory)[batch_mask]
                frames = torch.from_numpy(frames)[:, None, :].to(self.device).float()
                
                self.opt.zero_grad()

                y_pred = self.env_model(frames).to(self.device)
                y_target = self.random_net(frames).to(self.device)

                loss = criterion(y_pred, y_target)
                loss.backward()

                self.opt.step()
                self.lr_scheduler.step()

                if verbose:
                    print('Epoch {} - batch {}: Loss: {}'.format(
                           epoch + 1, batch_idx + 1, loss.item()))

    def update(self, stacked_frames):
        """ This function is called continously during the game to
        update the frame buffer and to train the environment model
        every `train_every` iterations.
        """
        self._memory.append(preprocess_frame(stacked_frames[-1]))
        
        if len(self._memory) and self.update_count % self.train_every == 0:
            self.train(batch_size=4, epochs=1, verbose=True)
        self.update_count += 1

    def distilled_dists(self, predicted_frames):
        """ Compute the distances between the student embeddings
        of the redicted frames and the teacher embeddings of 
        the predicted frames.
        """
        self.env_model.eval()
        criterion = self.get_criterion()
        dists = []

        with torch.no_grad():
            for i, predicted_frame in enumerate(predicted_frames):
                y_pred = self.env_model(predicted_frame[None, :]).to(self.device)
                y_target = self.random_net(predicted_frame).to(self.device)
                dists.append(torch.abs(criterion(y_pred, y_target)).item())

        return dists

    def get_action(self, stacked_frames):
        actions = torch.Tensor(np.eye(self.num_actions)).to(self.device)
        actions.to(self.device)

        # Get the predictions for the current frames.
        batch_frames = self.get_batch(stacked_frames).to(self.device)
        predicted_frames = self.model(batch_frames, actions)
        predicted_frames = predicted_frames.detach()

        if self.view_results:
            fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
            axes[0].imshow(stacked_frames[-1])
            axes[0].set_title('Last seen frame')

            idx_to_direction = {0: 'N', 1: 'S', 2: 'E', 3: 'W'}
            for i in range(4):
                axes[i + 1].imshow(predicted_frames[i][0].detach().cpu().numpy())
                axes[i + 1].set_title('Prediction for {}'.format(idx_to_direction[i]))
            plt.savefig('../imagination-results/{}.png'.format(self.cnt))
            plt.close()
            self.cnt += 1

        dists = self.distilled_dists(predicted_frames)
        dists = np.array(dists)
        dists /= np.sum(dists)
        
        best_action = np.random.choice(4, replace=False, p=dists)
        return best_action
