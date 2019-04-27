# PROGRAMMER:  Jeff Carter
# DATE CREATED:  2019-04-23
# REVISED DATE:  2019-04-26
# PURPOSE:  Trains a neural network to identify the type of flowers captured in an image

import util
import image_network
import torch
import checkpoint

from torch import nn, optim
from torchvision import models


def main():
    '''
    The main method of train.py
    '''   
    args = util.get_train_args()
    image_datasets, dataloaders = util.load_data(args.data_dir)
    
    if (args.cont):
        model, arch = checkpoint.load(args.cont)
    else:
        arch = args.arch
        model = image_network.create_model(arch, args.hidden_units, 2, 102)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    model.to(args.device)
    
    image_network.train(dataloaders, model, criterion, optimizer, args.epochs, args.device)
    
    if args.save_dir:
        checkpoint.save(arch, model, args.save_dir, image_datasets['train'].class_to_idx)

    
if __name__ == '__main__':
    main()