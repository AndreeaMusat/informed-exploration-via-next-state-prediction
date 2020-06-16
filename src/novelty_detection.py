import collections
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import operator

from abc import abstractmethod
from PIL import Image
from skimage import color

from unet_model_cp import (
    AdditiveConditionalUNet,
    ConcatConditionalUNet,
)
from unet_model_cp_new import UNetConcat

models = {'unet_concat' : ConcatConditionalUNet(n_channels=4, n_classes=1, action_channels=64)}


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
        frame = np.array(frame).astype(float) / 255.0
        frame = (frame - 0.65491969107) / 0.2275179
            
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



class EnvNet(nn.Module):
    def __init__(self):
        super(EnvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(57600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class RandomNet(nn.Module):
    def __init__(self):
        super(RandomNet, self).__init__()
        self.fc = nn.Linear(64 * 64, 10)

    def forward(self, x):
        with torch.no_grad():
            return self.fc(torch.flatten(x, 1))


class RandomDistillation(Imagination):
    """Imagination module via random distillation"""
    def __init__(self, num_actions, memory,
                 capacity, model_type, model_path, train_every=5):
        super(RandomDistillation, self).__init__()

        self.num_actions = num_actions

        # memory` is a list of previously seen frames.
        self._memory = collections.deque(memory, capacity)
        self.train_every = train_every
        self.train_count = 0
        self.device = 'cpu'

        self.model = models[model_type]
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval().to(self.device)

        self.env_model = EnvNet().to(self.device)
        self.opt = torch.optim.Adam(self.env_model.parameters(), lr=0.001)

        self.random_net = RandomNet().to(self.device)

    def preprocess_frame(self, frame):        
        frame = np.array(frame).astype(float) / 255.0
        frame = (frame - 0.65491969107) / 0.2275179

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


    def train(self, batch_size, epochs=2):
        self.env_model.train()
        criterion = torch.nn.MSELoss()

        print('Training environment model...')
        for epoch in range(epochs):
            for batch_idx in range(len(self._memory) // batch_size):
                batch_mask = np.random.choice(len(self._memory), batch_size, replace=False)
                frames = np.array(self._memory)[batch_mask]
                frames = torch.from_numpy(frames)[:, None, :].float()

                self.opt.zero_grad()

                y_pred = self.env_model(frames).to(self.device)
                y_target = self.random_net(frames).to(self.device)

                loss = criterion(y_pred, y_target)
                loss.backward()
                self.opt.step()

                print('Epoch {} - batch {}: Loss: {}'.format(epoch + 1, batch_idx + 1, loss.item()))

    def update(self, stacked_frames):
        self._memory.append(self.preprocess_frame(stacked_frames[-1]))
        
        if len(self._memory) >= 50 and self.train_count % self.train_every == 0:
            self.train(batch_size=16, epochs=2)
            self.train_count += 1

    def distilled_dists(self, predicted_frames):
        self.env_model.eval()
        criterion = torch.nn.MSELoss()
        dists = {}

        for i, predicted_frame in enumerate(predicted_frames):
            y_pred = self.env_model(predicted_frame[None, :]).to(self.device)
            y_target = self.random_net(predicted_frame).to(self.device)
            dists[i] = criterion(y_pred, y_target).item()

        return dists

    def get_action(self, stacked_frames):
        actions = torch.Tensor(np.eye(self.num_actions))
        actions.to(self.device)

        batch_frames = self.get_batch(stacked_frames).to(self.device)

        predicted_frames = self.model(batch_frames, actions)
        predicted_frames = predicted_frames.detach()

        dists = self.distilled_dists(predicted_frames)
        best_action = max(dists.items(), key=operator.itemgetter(1))[0]

        return best_action
