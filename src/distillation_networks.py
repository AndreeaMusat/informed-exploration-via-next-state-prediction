import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomNet(nn.Module):
    """ Teacher network in random distillation.
    
    This model produces fixed (untrainable) 512-dimensional
    embeddings of the frames.
    """
    def __init__(self):
        super(RandomNet, self).__init__()
        self.fc = nn.Linear(64 * 64, 512)

    def forward(self, x):
        with torch.no_grad():
            return self.fc(torch.flatten(x, 1))
        
        
class EnvNet(nn.Module):
    """ Student network in random distillation.
    
    This model produces 512-dimensional embeddings of
    the frames that are trained to be close to the 
    embeddings of the teacher network above.
    """
    def __init__(self):
        super(EnvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(57600, 2048)
        self.fc2 = nn.Linear(2048, 512)

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
