import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import *
nclasses = 37 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.nets = resnet152()
        self.nets.avgpool = nn.AvgPool2d(2,stride=1)
        self.nets.conv1 = nn.Conv2d(
            3*5, 64, kernel_size=3, stride=1, padding=1,
            bias=True)
        self.fc1 = nn.Linear(1000, 256)
        self.output = nn.Linear(256, nclasses)

    def forward(self, x):
        x = self.nets(x)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.sigmoid(self.output(x))
        return x
