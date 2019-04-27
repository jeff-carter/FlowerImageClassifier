# PROGRAMMER:  Jeff Carter
# DATE CREATED:  2019-04-23
# REVISED DATE:  2019-04-26
# PURPOSE:  Saves and loads neural network models

import torch

from torchvision import models


def save(arch, model, filepath, class_to_idx):
    '''
    Saves a checkpoint of a trained model
    Parameters:
     arch - The architecture of the model
     model - The model
     filepath - The file path to save the checkpoint to
     class_to_idx
    Returns
     None
    '''
    checkpoint = {
        'arch': arch,
        'classifier': model.classifier,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict()
    }
        
    torch.save(checkpoint, filepath)
    
    
def load(filepath):
    '''
    Loads a saved checkpoint of a trained model
    Parameters:
     filepath - The file path of the checkpoint to load
    Returns:
     model - The model loaded from the checkpoint
     arch - The architecture of the model
    '''
    checkpoint = torch.load(filepath)
    
    if (checkpoint['arch']=='vgg16'):
        model = models.vgg16(pretrained=True)
    elif (checkpoint['arch']=='vgg13'):
        model = models.vgg13(pretrained=True)
    elif (checkpoint['arch']=='alexnet'):
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('arch specified is not permitted')
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['arch']