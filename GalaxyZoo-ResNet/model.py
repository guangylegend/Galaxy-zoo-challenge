import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *
nclasses = 37 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = resnet152()
        self.resnet.avgpool = nn.AvgPool2d(2,stride=1)
        self.fc1 = nn.Linear(1000, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, nclasses)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), training=self.training)
        x = F.dropout(F.relu(self.fc3(x)), training=self.training) 
        x = F.sigmoid(10*self.output(x))
        return x
