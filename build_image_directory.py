import pandas as pd
from shutil import copyfile
from docopt import docopt
import os
from shutil import rmtree as delete_folder
from util import copy_all_images, add_three_categories
from constants import *


"""
Script used to create the splits for data: given a csv with images and classes and a directory of images
the script creates directories in the provided directories for both training and validation data and copies images into
those directories to allow easy creation of dataloaders for training/testing.


label_path: Path for csv with descretized labels, image names, scores, and city
train_dir: Directory to copy the training images into
test_dir: Directory to copy test images into
val_dir: Directory to copy validation images into
img_dir: Directory that contains all of the images
categories: How many labels are contained in the csv.  
test_split (optional): Fraction of images that are used for test dataset (default .1)
val_split (optional): Fraction of images that are used for val dataset (default .1)

example: 
python3 build_image_directory.py train --label_path model_data/resampled_train_binary1_full.csv --train_dir model_data/train --test_dir model_data/test 
--val_dir model_data/val --img_dir model_data/imgs  --categories 2  --test_split .1 --val_split .1

"""



usage = """Building image directory script

Usage:
    build_image_directory.py train --label_path <label_path> --train_dir <train_dir> --val_dir <val_dir> 
    --test_dir <test_dir> --img_dir <img_dir> --categories <categories> [--test_split <test_split>] [--val_split <val_split>]

"""





def split_and_copy_images(data, test_dir, val_dir ,train_dir, image_dir, test_split, val_split, categories=[0,1]):
    """
    For datasets that have not already been split into train and val splits.
    """
    def sample_and_copy_category(category):
        level_data = data[data.rating == category]

        val_total = int(val_split * len(level_data))
        test_total = int(test_split * len(level_data))
        level_val = level_data.sample(n= test_total)
        level_data  = level_data.drop(level_val.index)
        level_test = level_data.sample(n= val_total)
        level_train = level_data.drop(level_test.index)

        print('copying {} level {} validation images'.format(len(level_val), category))
        val_category_dir = os.path.join(val_dir)
        if not os.path.exists(val_category_dir):
            os.makedirs(val_category_dir)
       

        print('copying {} level {} test images'.format(len(level_test), category))
        test_category_dir = os.path.join(test_dir)
        if not os.path.exists(test_category_dir):
            os.makedirs(test_category_dir)


        print('copying {} level {} training images'.format(len(level_train), category))
        train_category_dir = os.path.join(train_dir)
        if not os.path.exists(train_category_dir):
            os.makedirs(train_category_dir)

        copy_images(level_train, level_val , level_test ,test_dir, val_dir, train_dir,image_dir, categories)

    for c in categories:
        sample_and_copy_category(c)


#This is an alternative function if you wish to add the data to the directories in a specific split.  
def copy_images(train_data, val_data, test_data, test_dir, val_dir, train_dir, image_dir, categories=[0,1]):
    
    def sample_and_copy_category(category):
        level_train = train_data[train_data.rating == category]
        level_val = val_data[val_data.rating == category]
        level_test = test_data[test_data.rating == category]

        print('copying {} level {} validation images'.format(len(level_val), category))
        val_category_dir = os.path.join(val_dir, str(category))
        if not os.path.exists(val_category_dir):
            os.makedirs(val_category_dir)
        copy_all_images(list(level_val.image_name), image_dir, val_category_dir)


        print('copying {} level {} test images'.format(len(level_test), category))
        test_category_dir = os.path.join(test_dir, str(category))
        if not os.path.exists(test_category_dir):
            os.makedirs(test_category_dir)
        copy_all_images(list(level_test.image_name), image_dir, test_category_dir)


        print('copying {} level {} training images'.format(len(level_train), category))
        train_category_dir = os.path.join(train_dir, str(category))
        if not os.path.exists(train_category_dir):
            os.makedirs(train_category_dir)
        copy_all_images(list(level_train.image_name), image_dir, train_category_dir)


    for c in categories:
        sample_and_copy_category(c)


#optional function that can be used to keep city directoes different
def copy_images_allcities(train_data, img_dir, res_dir,  categories=[0,1]):
    def sample_and_copy_category(category):
        level_train = train_data[train_data.rating == category]
        for city in CITIES:
            level_train_city = level_train[level_train.city == city]
            print('copying {} level {} training images for {}'.format(len(level_train_city), category, city))
            res_category_dir = os.path.join(res_dir,str(category))
            if not os.path.exists(res_category_dir):
                os.makedirs(res_category_dir)
            copy_all_images(list(level_train_city.image_name), img_dir, res_category_dir)

    for c in categories:
        sample_and_copy_category(c)





if __name__ == "__main__":

    arguments = docopt(usage)
    print(arguments)

    val_split = float(arguments['<val_split>'])  if arguments['--val_split'] else .1
    test_split = float(arguments['<test_split>'])  if arguments['--test_split'] else .1
    label_path = arguments['<label_path>']
    train_dir = arguments['<train_dir>']
    val_dir  = arguments['<val_dir>']
    test_dir = arguments['<test_dir>']
    img_dir = arguments['<img_dir>']
    categories = int(arguments['--categories'])

    print(f"label_dir: {label_path}")
    print(f"train_dir: {train_dir}")
    print(f"test_dir: {test_dir}")
    print(f"val_dir, : {val_dir}")
    print(f"img_dir: {img_dir}")
    print(f"test_split: {test_split}")
    print(f"test_split: {val_split}")
    print(f"categories: {categories}")
    categories = list(range(categories))
    # indicate all labels to use for split
    split_and_copy_images(pd.read_csv(label_path) , test_dir,val_dir, train_dir, img_dir, test_split, val_split, categories)





