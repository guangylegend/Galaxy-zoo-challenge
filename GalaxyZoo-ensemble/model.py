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
        self.net1 = resnet152()
        self.net1.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1,
            bias=True)
        self.net1.avgpool = nn.AvgPool2d(2,stride=1)

        self.net2 = resnet152()
        self.net2.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1,
            bias=True)
        self.net2.avgpool = nn.AvgPool2d(2,stride=1)

        self.net3 = resnet152()
        self.net3.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1,
            bias=True)
        self.net3.avgpool = nn.AvgPool2d(2,stride=1)

        self.net4 = resnet152()
        self.net4.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1,
            bias=True)
        self.net4.avgpool = nn.AvgPool2d(2,stride=1)

        self.net5 = resnet152()
        self.net5.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1,
            bias=True)
        self.net5.avgpool = nn.AvgPool2d(2,stride=1)
        
        self.fc1 = nn.Linear(5000, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.output = nn.Linear(256, nclasses)

    def forward(self, x):
        x = F.relu(torch.cat(
            [
                self.net1(x[:,0:3,:,:]),
                self.net2(x[:,3:6,:,:]),
                self.net3(x[:,6:9,:,:]),
                self.net4(x[:,9:12,:,:]),
                self.net5(x[:,12:15,:,:]),
            ], 1))
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), training=self.training)
        x = F.sigmoid(self.output(x))
        return x
