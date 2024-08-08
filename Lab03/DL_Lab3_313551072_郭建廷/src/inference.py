# inference.py: This file deals with model inference (making predictions) on unseen data. It
# includes functions to apply the trained model to new images.

import argparse
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import os.path
from models.resnet34_unet import *
from models.unet import *
from oxford_pet import *
import utils
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(model, dataset_loader, device,criterion,show_results = False):
    # implement the evaluation function here
    model.eval()
    total_loss = 0.0
    dice_score = 0.0
    for batch_idx, (images,masks) in enumerate(dataset_loader):
        images, masks = images.to(device),masks.to(device)
        #print(images.shape)
        outputs = model(images)
        pred_masks = F.sigmoid(outputs)
        #print(pred_masks,masks)
        if (show_results):
            utils.show_batch_masked_img(images,(pred_masks > 0.5))
        loss = criterion(pred_masks,masks)

        total_loss += loss.item()
        dice_score += utils.dice_score(pred_masks,masks).item()
    return total_loss / len(dataset_loader) , dice_score / len(dataset_loader) # loss & score


def load_model_weight(model, file_path):
    # Load the model state
    if os.path.isfile(file_path):
        state = torch.load(file_path)
        model.load_state_dict(state['model_state'])
        last_epoch = state['epoch']
        last_loss = state['loss']
        try:
            score = state['score']
        except:
            score = "NAN"
        print(f"Checkpoint loaded from {file_path} , last_epoch: {last_epoch} , last_loss: {last_loss} , dice_score: {score}")
        return last_epoch, last_loss
    else:
        print(f"No checkpoint file found at {file_path}")
        return 0, 0




def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default=None, help='path to the stored model weight')
    parser.add_argument('--model_name', default='unet', help='unet or resnet34_unet')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--show_inference', '-si', type=bool, default=False, help='show each image inference result')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    dataset = load_dataset(args.data_path ,"test")
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=args.batch_size,shuffle=False)
    model_name = ""
    if (args.model_name == "unet"):
        model = UNet(input_channel=3,output_channel=1).to(device) # binary semantic segmentation,
        model_name = "unet"
    else:
        model = ResNet34_UNet(input_channel=3,output_channel=1).to(device) 
        model_name = "resnet_unet"

    criterion = nn.BCEWithLogitsLoss()

    last_epoch,last_loss = 0,0
    if (args.model is not None):
        last_epoch,last_loss = load_model_weight(model,args.model)
    
    valid_loss,last_score = inference(model,dataset_loader,device,criterion,args.show_inference)
    print("loss:",valid_loss,"dice_score:",last_score)
    