# gsv-trash-replication-repo
This repository provides code, data, and training models to reproduce the SSO@S pipeline outlined in Hwang and Naik's (2023) paper, "[Systematic Social Observation at Scale: Using Crowdsourcing and Computer Vision to Measure Visible Neighborhood Conditions.](https://journals.sagepub.com/doi/abs/10.1177/00811750231160781)" This repository accompanies data and statistical code to replicate tables and figures in the manuscript provided [here](https://purl.stanford.edu/xy095yh6422).

Preferred Citation:
> Hwang, J. and Naik, N. (2023). Unrestricted data and statistical code accompanying Hwang, J. and N. Naik. 2023. "Systematic Social Observation at Scale: Using Crowdsourcing and Computer Vision to Measure Visible Neighborhood Conditions". Stanford Digital Repository. Available at https://purl.stanford.edu/xy095yh6422. https://doi.org/10.25740/xy095yh6422.


## Instructions for Set-up

Download Conda following the following online installation guide: 

https://conda.io/projects/conda/en/stable/user-guide/install/download.html

Setup Conda virtual environment for all needed dependencies with the following commands: 

`conda env create -f environment.yml`

(If on a M2 Mac use : 
`conda env create -f environment_mac.yml`
)

`conda activate gsv_trash`


## Python Scripts:

- constants.py: Specifies global constants such as trash trueskill thresholds and hyperparams. Additionally, it holds information that can be used to train the resnet classifier model. IE Information such as where to store the CSV file, where to find images, where to output CSV files, etc. 

- discretize_trueskill.py: Creates csv using inputted thresholds and raw trueskill scores to produce true labels.

- extract_vectors.py: Uses resnet model to produce feature embeddings of images.

- build_image_directory.py: Given a directory of images and a csv of images and their labels, splits and copies into new folders based on true labels for resnet training.

- trainer.py: Defines Trainer class that is utilized for training, checkpointing, evaluating, logging, and creating metrics for the resnet classifier training process.

- train.py: Initiates trainer and data loaders utilized for the training process and begins the training process for the resnet.

- image2vec.py: Class to convert images to vector embeddings used to train/test SVMs, uses trained Resnet Classifier to create the embeddings . Used to create CSV of columns: Image name, renet prediction, embedding, and label to be used in svm_classifier.py

- model.py: Defines the Resnet backbone classifier model 

- svm_classifier.py: Given csv with image feature vectors and associated true labels, trains an SVC (or SVR if specified).

- test_model.py: Suite of methods to help with error analysis/model testing

- util.py: Provides miscellaneous helper functions

## Training Pipeline:

_Inputs_: image_dir (directory with all images), trueskill_csv (a csv that contains image_name and associated score)

1. use discretize_trueskill.py using the trueskill_csv produce a csv containing image_name and true label

2. run build_image_directory.py use image_dir and discretize_trueskill.py output to create labeled image directories to be used for training

3. run train.py to use the labeled images from the previous step to train a classifier model with resnet

_(To read the evaluation metrics during the training process use the following command: tensorboard --logdir <LOG_DIR>)_

4. run extract_vectors.py utilizes a trained classifier model to extract training and test image vectors

5. run svm_classify.py to utilize extracted feature vectors to train/test an SVM model

6. run Trash_analysis.ipynb to make an analysis of the final trained classifier model. 

_Outputs_:  Resnet Classifier/Feature Extractor, Feature extractions of the images, Trained SVM classifier on Feature extractions, Tensorboard logs

# Correspondence
Contact Jackelyn Hwang at jihwang@stanford.edu

# License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
