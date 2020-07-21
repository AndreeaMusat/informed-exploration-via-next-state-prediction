import torch
import torch.nn as nn


class LinearDiscriminator(nn.Module):
    def __init__(self, input_size):
        super(LinearDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
    
    
class ConvDiscriminator(nn.Module):
    """ Discriminator conditioned on both the action and previous frames.
    """
    def __init__(self, in_channels=1):
        super(ConvDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 32, normalization=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 1, 4, padding=1, bias=False)
        )
        self.linear = nn.Linear(16, 1)

    def forward(self, img_A, action):
        # Concatenate image and condition image by channels to produce input.
        action_maps = torch.ones_like(img_A) * action[:, None, None, None].float()
        img_input = torch.cat((img_A, action_maps), 1)
        model_output = torch.flatten(self.model(img_input), 1).reshape(img_A.shape[0], -1)
        
        # Predict the probability that this is the correct next
        # frame given the action and previous frames.
        return torch.sigmoid(self.linear(model_output))
