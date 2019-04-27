# PROGRAMMER:  Jeff Carter
# DATE CREATED:  2019-04-23
# REVISED DATE:  2019-04-26
# PURPOSE:  Creates and trains neural networks

import torch

from torchvision import models
from torch import nn

def create_model(arch, hidden_units, hidden_layers, output_size):
    '''
    Creates a neural network model
    Parameters:
     arch - the architecture of the model
     hidden_units - the number of units in each hidden layer
     hidden_layers - the number of hidden layers in the model
     output_size - the number of units in the output layer
    Returns:
     model - the resulting model
    '''
    model, classifier = None, None
    
    if (arch=='vgg16'):
        model = models.vgg16(pretrained=True)
        classifier = create_vgg_classifier(hidden_units, hidden_layers, output_size)
    elif (arch=='vgg13'):
        model = models.vgg13(pretrained=True)
        classifier = create_vgg_classifier(hidden_units, hidden_layers, output_size)
    elif (arch=='alexnet'):
        model = models.alexnet(pretrained=True)
        
        layers = list()
        layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(9216, hidden_units))
        layers.append(nn.ReLU())
        
        for idx in range(hidden_layers):
            append_hidden_layer(layers, 0.1, hidden_units)
        
        layers.append(nn.Linear(hidden_units, output_size))
        layers.append(nn.LogSoftmax(dim=1))
        
        classifier = nn.Sequential(*layers)
    else:
        raise ValueError('arch specified is not permitted')
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = classifier
        
    return model

def create_vgg_classifier(hidden_units, hidden_layers, output_size):
    '''
    Creates a classifier for a VGG model
    Parameters:
     hidden_units - the number of units in each hidden layer
     hidden_layers - the number of hidden layers in the model
     output_size - the number of units in the output layer
    Returns:
     classifier - the classifier that has been created
    '''
    layers = list()
    layers.append(nn.Dropout(p=0.1))
    layers.append(nn.Linear(25088, hidden_units))
    layers.append(nn.ReLU())

    for idx in range(hidden_layers):
        append_hidden_layer(layers, 0.1, hidden_units)

    layers.append(nn.Linear(hidden_units, output_size))
    layers.append(nn.LogSoftmax(dim=1))

    classifier = nn.Sequential(*layers)
        
    return classifier

def append_hidden_layer(layer_list, dropout, hidden_features):
    '''
    Appends a hidden layer to a list of layers
    Parameters:
     layer_list - a list of network layers, should already have the input layer present
     dropout - the rate of dropout to use in the layer
     hidden_features - the number of units to have in the layer
    Returns:
     None
    '''
    layer_list.append(nn.Dropout(p=dropout))
    layer_list.append(nn.Linear(hidden_features, hidden_features))
    layer_list.append(nn.ReLU())
    

def train(dataloaders, model, criterion, optimizer, epochs, device):
    '''
    Trains a neural network model
    Parameters:
     dataloaders - The dataloaders that will be used for both training data and validation data
     model - The model that will be trained
     criterion - The criterion
     optimizer - The optimizer
     epochs - The number of training epochs
     device - The type of device to use during training, e.g., cpu, cuda
    Returns
     None
    '''
    print_period = 10

    train_loss = 0
    steps = 0

    for epoch in range(epochs):
        print(f"Training... Epoch {epoch+1} of {epochs}...")
        
        for inputs, labels in dataloaders['train']:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Track the loss and accuracy on the validation set to determine the best hyperparameters
            if steps % print_period == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)

                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Step: {steps}\t"
                      f"Training loss: {train_loss/print_period:.3f}\t"
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}\t"
                      f"Validation Accuracy: {accuracy/len(dataloaders['valid'])*100:.3f}%")

                train_loss = 0
                model.train()
                
    print("Training Complete")