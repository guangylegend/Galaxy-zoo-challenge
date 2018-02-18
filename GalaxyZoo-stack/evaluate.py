from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets

from data import initialize_data # data.py in the same folder
from model import Net

parser = argparse.ArgumentParser(description='PyTorch Galaxy Zoo evaluation script')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='galaxyzoo_pred.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()

state_dict = torch.load(args.model)

cuda = torch.cuda.is_available()

model = Net()
model.load_state_dict(state_dict)

if cuda:
    model = model.cuda()
model.eval()

from data import data_transforms_val

test_dir = args.data + '/images_test_rev1'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(args.outfile, "w")
output_file.write(
    "GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,"
    "Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,"
    "Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,"
    "Class11.2,Class11.3,Class11.4,Class11.5,Class11.6\n")
for f in tqdm(os.listdir(test_dir)):
    if f.endswith('.jpg'):
        data = data_transforms_val(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        data = Variable(data, volatile=True)
        if cuda:
            data = data.cuda()
        output = model(data)
        pred = output.data[0]
        file_id = f[:-4]
        output_file.write("%s,%s\n" % (file_id, ",".join([str(x) for x in pred.cpu().numpy()])))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle '
      'competition at https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge')
        


