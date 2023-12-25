"""Extract Image Features from resnet classifier backbone and store in csv.

Usage:
    extract_image_features.py  --output_csv_path <output_path> --image_dir <image_dir> --num_output_labels <num_labels> --model_path <model>
                              [--is_train <train_flag>]

Options:
    --output_csv_path <output_path>   Path to output image names and their extracted features
    --image_dir <image_dir>           Folder containing images
    --num_output_labels <num_labels>  Int specifying number of categories (e.g., 2 for binary, 4 for multiclass) [default: 2]
    --model_path <model>              Path to Resnet model to use for feature extraction
    --is_train <train_flag>           Bool indicating whether it is a training run [default: True]
"""

#Example run:
#python extract_vectors.py --output_csv_path ./temp.csv --image_dir ./model_data/train --num_output_labels 2 --model_path ./model_data/10000.pth --is_train True


import warnings
import json


import pandas as pd
from PIL import Image
from docopt import docopt
import torchvision.datasets as datasets

import warnings
import json

from model import Classifier


import os
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torchvision import datasets, models, transforms

from svm_classify import train_model, predict_test

import numpy as np

import image2vec


warnings.filterwarnings("ignore")


BATCH_SIZE = 64
NUM_WORKERS = 4

 
#Transformations for training, adds some color jitter
TRAIN_TRANSFORMATION = transforms.Compose([
           transforms.RandomResizedCrop(224),
           transforms.RandomHorizontalFlip(),
           transforms.ColorJitter(brightness=.05),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#Transformation for infrence, no color jitter
TEST_TRANSFORMATION = transforms.Compose([
           transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ImageFolderWithPaths(datasets.ImageFolder):
    """
    defined ImageFolder class that allows for the dataloader to also output the img name when iterating through
    """
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def extract_new(num_output_labels, model_path, img_path, output, is_train = True):
    """

    Args:
        num_output_labels: output classes of the trained model
        model_path: Path of the trained model
        img_path: Path with the images to extract from
        output: output csv name
        is_train: used to determine test/train transformations for the dataloader

    creates csv with features, labels, image name and classifier predictions

    """
    img_2_vec = image2vec.Img2Vec(model= model_path, num_output_labels=num_output_labels)
    data_transform = TRAIN_TRANSFORMATION if is_train == "train" else TEST_TRANSFORMATION
    image_dataset = ImageFolderWithPaths(img_path, data_transform)
    data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    img_2_vec.get_vec(data_loader, output_name= output, add_name =True)

if __name__=="__main__":
    args = docopt(__doc__)
    print(args)
    output_name = args['--output_csv_path']
    image_directory = args['--image_dir']
    num_output_labels = args['--num_output_labels']
    model_path = args['--model_path']
    is_train = args ['--is_train']
    extract_new(num_output_labels, model_path, image_directory , output_name , is_train)