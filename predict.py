# PROGRAMMER:  Jeff Carter
# DATE CREATED:  2019-04-23
# REVISED DATE:  2019-04-26
# PURPOSE:  Uses a trained neural network to identify the type of flowers captured in an image

import util
import checkpoint
import numpy as np
import torch
import json

from PIL import Image


def main():
    '''
    The main method of train.py
    '''    
    args = util.get_predict_args()
    
    np_image = process_image(args.path_to_image)
    
    if (args.device == 'cuda'):
        t_image = torch.from_numpy(np_image).type(torch.cuda.FloatTensor).unsqueeze(0)
    else:
        t_image = torch.from_numpy(np_image).type(torch.FloatTensor).unsqueeze(0)
    
    model = checkpoint.load(args.checkpoint)[0]
    model.to(args.device)
    
    model.eval()
    with torch.no_grad():
        logps = model.forward(t_image)
        ps = torch.exp(logps)
    
    top_p, top_class = ps.topk(args.top_k, dim=1)
    np_top_p, np_top_class = top_p.cpu().numpy(), top_class.cpu().numpy()
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            
        for (x, y), value in np.ndenumerate(np_top_class):
            print(f"Name: {cat_to_name.get(str(value))},\tCertainty: {np_top_p[x][y]:.5f}")
        
    else:    
        for (x, y), value in np.ndenumerate(np_top_class):
            print(f"Class: {value},\tCertainty: {np_top_p[x][y]:.5f}")
    
    
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        Parameters:
         image_path - the file path of the image
        Returns:
         np_image - the processed image as a numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    
    image.thumbnail((256, 256))
    width, height = image.size
    
    new_width, new_height = 224, 224
    
    left = (width - new_width)/2
    upper = (height - new_height)/2
    right = left + new_width
    lower = upper + new_height
    
    image = image.crop((left, upper, right, lower))
    
    
    mean_colors = np.array([0.485, 0.456, 0.406])
    std_dev_colors = np.array([0.229, 0.224, 0.225])
    
    np_image = np.array(image) / 255
    
    np_image = (np_image - mean_colors) / std_dev_colors
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image
    
    
if __name__ == '__main__':
    main()