#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import torch.utils.data

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from discriminators import (
    LinearDiscriminator,
    ConvDiscriminator
)

from gradient_difference_loss import *
from maze_dataset import MazeDataset

from unet_models import (
    AdditiveConditionalUNet,
    ConcatConditionalUNet
)

import sys
print(sys.version)
sys.executable
print(torch.cuda.is_available())


dataloaders = {}
for mode in ['train', 'valid', 'test']:
    dataset = MazeDataset(
        root_dir='../maze_data_full/maze_data',
        mode=mode,
        in_memory=False,
    )
    
    dataloaders[mode] = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=32, 
        shuffle=True, 
    )
    
    print('Num batches %s: %d' % (mode, len(dataloaders[mode])))


def print_result(last_img, true_img, pred_img, action, epoch, batch_idx, msg, show=False):
    frame_idx = np.random.choice(curr_states.shape[0])
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.tight_layout()
    
    axes[0].imshow(last_img.cpu().numpy())
    axes[1].imshow(true_img.cpu().numpy()[0])            
    axes[2].imshow(pred_img.detach().cpu().numpy()[0])

    axes[0].set_title('{} Last seen frame, action {} '.format(msg, action))
    axes[1].set_title('{} True next frame'.format(msg))
    axes[2].set_title('{} Pred next frame epoch {} batch {}'.format(msg, epoch, batch_idx))

    plt.savefig('../predictions-conv-gan/{}-action-{}-epoch-{}-batch-{}.png'.format(msg, action, epoch, batch_idx))
    
    if show:
        plt.show()
    plt.close()


# Loss function
adversarial_loss = torch.nn.BCELoss().cuda()
pixel_criterion = torch.nn.L1Loss().cuda()

discriminator  = ConvDiscriminator().cuda()
generator = ConcatConditionalUNet(n_channels=4, 
                                  n_classes=1, 
                                  action_channels=100).cuda()

adv_factor = 0.05
l1_factor = 1.0

lambda_pixel = 100
idx_to_direction = {0: 'N', 1: 'S', 2: 'E', 3: 'W'}


Tensor = torch.cuda.FloatTensor
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

num_epochs = 20
checkpoint_every = 50
show_results_every = 5
num_epochs = 10


for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    idx = 0
    for (batch_idx, batch) in enumerate(dataloaders['train']):
        idx += 1
        
        curr_states = batch['curr_state'].float().cuda()
        actions = batch['action'].float().cuda()
        actions_idxes = torch.argmax(actions, 1)
        next_frames = batch['next_frame'].float().cuda()
        weights = batch['weights'].float().cuda()

        valid = Variable(Tensor(next_frames.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(next_frames.size(0), 1).fill_(0.0), requires_grad=False)    

        optimizer_G.zero_grad()
        gen_next_frames = torch.tanh(generator(curr_states, actions))
        
        pixel_loss = pixel_criterion(gen_next_frames, next_frames)
        pixel_loss = torch.mul(1, pixel_loss).mean() 
        grad_loss = gradient_difference_loss(gen_next_frames, next_frames, reduction='none')
        grad_loss = torch.mul(1, grad_loss).mean()
        
        g_loss = adv_factor * adversarial_loss(discriminator(gen_next_frames, actions_idxes), valid) 
        g_loss += l1_factor * pixel_loss + grad_loss

        g_loss.backward()
        optimizer_G.step()

        if idx % 10 != 0:
            print('Epoch {}, Batch{}, Generator loss {}'.format(
           epoch, batch_idx, g_loss.item()))
        
        if idx % 10 == 0:
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(next_frames, actions_idxes), valid)
            fake_loss = adversarial_loss(discriminator(gen_next_frames.detach(), actions_idxes), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print('Epoch {}, Batch{}, Generator loss {}, Discriminator loss {}'.format(
            epoch, batch_idx, g_loss.item(), d_loss.item()))

            if batch_idx % checkpoint_every == 0:
                print('Saving model..')
                checkpoint_name = '../models/gan-checkpoint-loss{:.5f}-epoch{}-batch{}'
                torch.save(
                    generator.state_dict(), 
                    checkpoint_name.format(
                        g_loss.item(), 
                        epoch,
                        batch_idx
                    )
                )
        
       
        if batch_idx % show_results_every == 0:
            frame_idx = np.random.choice(curr_states.shape[0])
            action = idx_to_direction[torch.argmax(actions[frame_idx]).item()]
            last_img = curr_states[frame_idx, -1]
            true_img = next_frames[frame_idx] 
            pred_img = gen_next_frames[frame_idx]
            print_result(last_img, true_img, pred_img, action, epoch, batch_idx, 'train')

