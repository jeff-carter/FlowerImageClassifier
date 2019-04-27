# PROGRAMMER:  Jeff Carter
# DATE CREATED:  2019-04-23
# REVISED DATE:  2019-04-26
# PURPOSE:  Contains utility functions that are used by train.py and predict.py

import argparse
import torch
import sys

from torchvision import transforms, datasets
from os import path


def get_train_args():
    '''
    Retrieves the arguments used by train.py
    Parameters:
     None
    Returns:
     args - Object of parsed command line arguments used in train.py
    '''
    default_arch = 'vgg16'
    default_learning_rate = 0.003
    default_hidden_units = 1024
    default_epochs = 5
    
    parser = argparse.ArgumentParser(description='Accepts several options')
    
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', type=str, default=None,
                        help="Use to set directory to save checkpoint")
    parser.add_argument('--arch', type=str, default=default_arch,
                        help="CNN Model Architecture with default value of '{}'".format(default_arch))
    
    parser.add_argument('--learning_rate', type=float, default=default_learning_rate,
                        help="Specify a learning rate for the CNN.  Defaults to {}".format(default_learning_rate))
    parser.add_argument('--hidden_units', type=int, default=default_hidden_units,
                        help="Specify the number of hidden units in the CNN.  Defaults to {}".format(default_hidden_units))
    parser.add_argument('--epochs', type=int, default=default_epochs,
                        help="Specify the number of epoch used during training.  Defaults to {}".format(default_epochs))
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu',
                        help="Use gpu for training instead of cpu")
    parser.add_argument('--cont', type=str, default=None,
                        help="Specify a checkpoint to continue training from")
    
    args = parser.parse_args()
    validate_train_args(args)
    
    return args


def validate_train_args(args):
    if args.cont:
        check_file_exists(args.cont)
    
    if args.device:
        check_gpu_available()
    
    if args.save_dir and path.isfile(args.save_dir):
        resp = None
        while resp not in ['Y', 'N']:
            resp = input("Checkpoint already exists.  Do you wish to overwrite? [Y/N]").upper()
            if resp=='N':
                sys.exit(0)


def get_predict_args():
    '''
    Retrieves the arguments for predict.py
    Parameters:
     None
    Returns:
     args - Object of parsed command line arguments used in predict.py
    '''
    default_top_k = '1'
    
    parser = argparse.ArgumentParser(description='Accepts several options')
    
    parser.add_argument('path_to_image')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k', type=int, default=default_top_k,
                        help="Number of top most likely classes to return.  Defaults to {}".format(default_top_k))
    parser.add_argument('--category_names', default=None,
                        help="JSON file that describes the mapping of catagory indexes to names")
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu',
                        help="Use gpu for training instead of cpu")
    
    args = parser.parse_args()
    validate_predict_args(args)
    
    return args


def validate_predict_args(args):
    check_file_exists(args.path_to_image)
    check_file_exists(args.checkpoint)
    
    if args.category_names:
        check_file_exists(args.category_names)
    
    if args.device:
        check_gpu_available()
    

def load_data(data_dir):
    '''
    Loads and organizes the sets of data used for training, validating, and testing
    Parameters:
     data_dir - The file path of the directory containing the sets of data
    Returns:
     image_datasets - The ImageFolder data sets
     dataloaders - The dataloaders
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    image_input_size = 224
    batch_size = 32

    norm_transform = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(image_input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm_transform
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(image_input_size),
            transforms.ToTensor(),
            norm_transform
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['test']),
        'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True)
    }
    
    return image_datasets, dataloaders
    
    
def check_file_exists(filepath):
    if not path.isfile(filepath):
        raise ValueError( "No such file as {}".format(filepath) )

        
def check_gpu_available():
    if not torch.cuda.is_available():
        raise ValueError("GPU is not available for use")
