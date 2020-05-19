import torch
from torch import nn


class QNetwork(nn.Module):
  
  def __init__(self, config):
    super(QNetwork, self).__init__()
    self.num_actions = config['num_actions']
    self.input_shape = config['input_shape']

    self.features = nn.Sequential(
      nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
      nn.ReLU(),
    )
    
    in_fc_size = self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    self.fc = nn.Sequential(
      nn.Linear(in_fc_size, 256),
      nn.ReLU(),
      nn.Linear(256, self.num_actions)
    )

    for layer in self.features:
      if hasattr(layer, 'weight'):
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
    for layer in self.fc:
      if hasattr(layer, 'weight'):
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')


  def forward(self, X):
    return self.fc(self.features(X).view(X.shape[0], -1))
