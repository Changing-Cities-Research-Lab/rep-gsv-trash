from trainer import Trainer
from model import Classifier
import torch
import torch.optim as optim
import torch.nn as nn
import random

from docopt import docopt
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torchvision import datasets, models, transforms
from sklearn.model_selection import KFold as KFold
import torchmetrics




import os

"""

Script utilized to initialize training/validation of the resnet backbone classifier: 

--train_path: Path of training images setup with build_image_directory.py script
--val_path: Path of training images setup with build_image_directory.py script
--log_path: Path to store graphs/confusion matrix created during the training process. 
--ckpt_path: Path to store checkpoints of the classifier during the training process.
--max_steps: Maximum amount of steps to train the classifier for
--eval_every: How many steps before classifier calculates metrics and creates confusion matrix on validation set
--ckpt_every: How often the model should create a checkpoint of itself. 

example use:

 python3 train.py --train_path model_data/train --val_path model_data/val --log_path model_data/log  --ckpt_path model_data/ckpt --num_classes 2 --max_steps 10000 --eval_every 1000 --ckpt_every 5000 

"""

usage = """Running the training

Usage:
    train.py --train_path <train_path> --val_path <val_path> --log_path <log_path>  --ckpt_path <ckpt_path>  --num_classes <num_classes> --max_steps <max_steps> --eval_every <eval_every> --ckpt_every <ckpt_every> 

"""


SEED = 42 # line for debugging reproducability
BATCH_SIZE =  64
NUM_WORKERS = 8

def init_trainer(loss_func, dataloaders, model, log_path, ckpt_path, device):
    """

    Args:
        loss_func: function for calculating loss.
        dataloaders: dataloaders for both training and validation
        model: the initialized classification model
        log_path: path for logging results
        ckpt_path: poth for saving checkpoints

    Returns:
        returns an initialized trainer class, as defined in the trainer.py file.

    """


    opt = optim.AdamW(model.parameters())
    #sch = lr_scheduler.StepLR(opt, step_size=10, gamma=0.8)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10000 , eta_min=0, last_epoch=- 1, verbose=False)

    c_trainer = Trainer(model,
                      opt,
                      sch,
                      loss_func,
                      dataloaders["train"],
                      dataloaders["val"],
                      log_path,
                      ckpt_path,
                      device)

    return c_trainer





def init_data(train_path, val_path):
    """
    Args:
        val_path: Path created using the build image directory script containing validation data
        train_path: Path created using the build image directory script containing training data

    Returns:
        returns the initialized data_loaders necessary for the training and eval process.
    """


    dirs = {
        'train': train_path,
        'val': val_path,
    }

    data_transforms = {
       'train': transforms.Compose([
           transforms.RandomResizedCrop(224),
           transforms.RandomHorizontalFlip(),
           transforms.ColorJitter(brightness=.05),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ]),
       'val': transforms.Compose([
           transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ])
    }



    image_datasets = {x: datasets.ImageFolder(dirs[x],
                                              data_transforms[x])
                      for x in ['train', 'val']}

    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=NUM_WORKERS)
                   for x in ['train', 'val']}


    return data_loaders, image_datasets, data_transforms



def check_dir(path):
    """
    Simply checks if the directory exists, and if it does not, creates a new directory at the given path
    """
    if not os.path.isdir(path):
        os.mkdir(path)
def seed_everything(seed=73):
    """

    Args:
        seed: value to seed

    updates env to keep everything as deterministic as possible for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = docopt(usage)
    print(args)
    paths = [args['<train_path>'], args['<val_path>'],args['<ckpt_path>'],args['<log_path>']]

    for path in paths:
        check_dir(path)
    assert os.path.isdir(args['<train_path>'])
    assert os.path.isdir(args['<val_path>'])
    assert os.path.isdir(args['<ckpt_path>'])
    assert os.path.isdir(args['<log_path>'])

    num_classes = args['<num_classes>']
    log_path = args['<log_path>']
    ckpt_path = args['<ckpt_path>']
    ckpt_every = args['<ckpt_every>']
    eval_every = args['<eval_every>']
    max_steps = args['<max_steps>']

    #above lines get values from the input creates directories for the directories that don't exist.
    

    #seed for reproducibility
    seed_everything(123)

    #Sets the device for the training. If GPU exists does it on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Creates the Dataloaders and Datasets needed to train the model
    dataloaders, datasets, _ =  init_data(args['<train_path>'], args['<val_path>'])

    #creates a final loss function for training. Weight is an optional setting to make different classes weigh more in the training.
    loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([1,2],dtype=torch.float).to(device),  reduction='mean')

    #creates an instance of the model and moves it to the device to start training 
    model = Classifier(int(num_classes)).to(device)

    #creates an instance of a trainer class defined in trainer.py to do the training
    trainer = init_trainer(loss_func, dataloaders, model, log_path, ckpt_path,device)

    #starts the training with the retrieved inputs.
    trainer.train(int(max_steps), int(eval_every), int(ckpt_every))
