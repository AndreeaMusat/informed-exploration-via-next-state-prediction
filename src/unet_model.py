# Adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts_cp import *


class AdditiveConditionalUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        num_channels = [32, 64, 128, 256]
     
        self.inc = DoubleConv(n_channels, num_channels[0])   
        self.down1 = Down(num_channels[0], num_channels[1])
        self.down2 = Down(num_channels[1], num_channels[2])
        self.down3 = Down(num_channels[2], num_channels[3] // factor)
        self.linear = nn.Linear(4, 16384 // factor)

        self.up2 = Up(num_channels[3], num_channels[2] // factor, bilinear)
        self.up3 = Up(num_channels[2], num_channels[1] // factor, bilinear)
        self.up4 = Up(num_channels[1], num_channels[0], bilinear)
        self.outc = OutConv(num_channels[0], n_classes)

    def forward(self, x, action):
        batch_size = x.shape[0]
        
        # Forward through the contracting path.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Additive interactions with the action.
        bottleneck_shape = x4.shape
        x4 = x4.view(batch_size, -1) + self.linear(action)
        x4 = x4.view(bottleneck_shape)
        
        # Forward through the expanding path.
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ConcatConditionalUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetConcat, self).__init__()
        num_channels = [32, 64, 128, 256]
        factor = 2 if bilinear else 1
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.action_channels = 100
        
        self.inc = DoubleConv(n_channels, num_channels[0]) 
        self.down1 = Down(num_channels[0], num_channels[1])
        self.down2 = Down(num_channels[1], num_channels[2])
        self.down3 = Down(num_channels[2], num_channels[3] // factor - self.action_channels)

        num_actions = 4
        self.linear = nn.Linear(num_actions, self.action_channels * 8 * 8)
        self.up2 = Up(num_channels[3], num_channels[2] // factor, bilinear)
        self.up3 = Up(num_channels[2], num_channels[1] // factor, bilinear)
        self.up4 = Up(num_channels[1], num_channels[0], bilinear)
        self.outc = OutConv(num_channels[0], n_classes)


    def forward(self, x, action):
        batch_size = x4.shape[0]
        
        # Forward through the contracting path.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Concatenate the action embedding with the bottleneck.
        action_embedding = self.linear(action)
        act_emb = torch.nn.functional.relu(action_embedding)
        act_emb = act_emb.view(-1, self.action_channels, original_shape[2], original_shape[3])
        
        # Forward through the expanding path.
        x = torch.cat((x4, act_emb), 1)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
