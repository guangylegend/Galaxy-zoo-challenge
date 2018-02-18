from __future__ import print_function

import re
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from model import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_file', type=str, default=None, metavar='PU',
                    help='pick up where you were (default: None)')
parser.add_argument('--model_name', type=str, default='model', metavar='MN',
                    help='name of the model file (default: model)')
args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
train_images, train_labels, val_images, val_labels = initialize_data(args.data) # extracts the zip files, makes a validation set




train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size, shuffle=False, num_workers=4)

cuda = torch.cuda.is_available()
net = Net()

first_epcoh = 1

if args.model_file and os.path.isfile(args.model_file):
    print("loading {}".format(args.model_file))
    first_epcoh = int(re.findall('_(\d+).pth', args.model_file)[0]) +1
    net.load_state_dict(torch.load(args.model_file))

cuda = torch.cuda.is_available()
if cuda:
    net = net.cuda()

# optimizer = optim.Adadelta(net.parameters())
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, threshold=0.001)

def rmse_loss(input, target, size_average=True):
    """Compute root mean squared error"""
    loss = torch.sqrt(
            torch.mean((input - target).pow(2), 1))
    if size_average:
        return torch.mean(loss)
    else:
        return torch.sum(loss)

LossFunc = rmse_loss

def train(epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = net(data)
        loss = LossFunc(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def validation():
    net.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = net(data)
        lo = LossFunc(output, target, size_average=False).data[0] # sum up batch loss
        validation_loss += lo
        # print(lo)

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}\n'.format(validation_loss))
    return validation_loss

ploss = 0
pploss = 0
val_loss = 0
loss_min = 1000
PreState = None
strict_save = args.model_file is not None

for epoch in range(first_epcoh, args.epochs + 1):
    PreState = net.state_dict()
    train(epoch)
    pploss, ploss, val_loss = ploss, val_loss, validation() 
    # scheduler.step(val_loss)
    if pploss > ploss and ploss < val_loss and (not strict_save or strict_save and ploss < loss_min):
        loss_min = ploss
        model_file = "{}_{}.pth".format(args.model_name, epoch-1)
        torch.save(PreState, model_file)
        print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
        strict_save = True
        
model_file = "{}_{}.pth".format(args.model_name, args.epochs)
torch.save(PreState, model_file)
