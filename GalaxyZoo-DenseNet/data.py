from __future__ import print_function
import zipfile
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import PIL.Image as Image


# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 224 x 224 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.Scale((96, 96)),
    transforms.RandomCrop((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.0930514,0.08065233,0.06301059), (0.13712655,0.11481636,0.09914573)),
    ])

data_transforms_val = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.Scale((96, 96)),
    transforms.CenterCrop((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.0930514,0.08065233,0.06301059), (0.13712655,0.11481636,0.09914573)),
    ])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def initialize_data(folder):
    train_zip = folder + '/images_training_rev1.zip'
    test_zip = folder + '/images_test_rev1.zip'
    label_zip = folder + '/training_solutions_rev1.zip'
    if (not os.path.exists(train_zip) or
        # not os.path.exists(test_zip) or 
        not os.path.exists(label_zip)):
        raise(RuntimeError("Could not find " + train_zip + " and " + test_zip
            + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2017/data '))

    # extract images_training_rev1.zip to images_training_rev1
    train_folder = folder + '/images_training_rev1'
    if not os.path.isdir(train_folder):
        print(train_folder + ' not found, extracting ' + train_zip)
        zip_ref = zipfile.ZipFile(train_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()

    # extract images_test_rev1.zip to images_test_rev1
    test_folder = folder + '/images_test_rev1'
    if not os.path.isdir(test_folder):
        print(test_folder + ' not found, extracting ' + test_zip)
        zip_ref = zipfile.ZipFile(test_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()

    # extract training_solutions_rev1.zip to training_solutions_rev1.csv
    label_file = folder + '/training_solutions_rev1.csv'
    if not os.path.isfile(label_file):
        print(label_file + ' not found, extracting ' + label_zip)
        zip_ref = zipfile.ZipFile(label_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()

    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + '/val_images'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):

            if dirs[1] == '1':
                # move file to validation folder
                os.rename(train_folder + '/' + dirs, val_folder + '/' + dirs )

    val_file = folder + "/val_label.csv"
    train_file = folder + "/train_label.csv"
    if not os.path.isfile(val_file) or not os.path.isfile(train_file):
        with open(label_file) as f:
            f.next()
            with open(val_file, 'w') as fval, open(train_file, 'w') as ftrain:
                for lines in f:
                    if lines[1] == '1':
                        fval.write(lines)
                    else:
                        ftrain.write(lines)
    
    # Read training and validation labels
    train_labels_raw = np.loadtxt(train_file, delimiter=',')
    val_labels_raw = np.loadtxt(val_file, delimiter=',')
    
    # Convert the possibility vector
    train_labels_id = train_labels_raw[:,0]
    val_labels_id = val_labels_raw[:,0]
    
    train_labels_tensor = torch.FloatTensor(train_labels_raw[:,1:])
    val_labels_tensor = torch.FloatTensor(val_labels_raw[:,1:])

    print("Making training set...")
    train_images = []
    for idx in train_labels_id:
        data = data_transforms(pil_loader("{}/{}.jpg".format(train_folder, int(idx))))
        train_images.append(data.view(1, data.size(0), data.size(1), data.size(2)))
    train_images_tensor = torch.cat(train_images)
    print("Making validation set...")
    val_images = []
    for idx in val_labels_id:
        data = data_transforms_val(pil_loader("{}/{}.jpg".format(val_folder, int(idx))))
        val_images.append(data.view(1, data.size(0), data.size(1), data.size(2)))
    val_images_tensor = torch.cat(val_images)

    return train_images_tensor, train_labels_tensor, val_images_tensor, val_labels_tensor

if __name__ == '__main__':
    tensors = initialize_data('data')
    for t in tensors:
        print(t.shape)
    training = tensors[0].numpy()
    print(np.mean(training,axis=(0,2,3)))
    print(np.std(training,axis=(0,2,3)))
    
