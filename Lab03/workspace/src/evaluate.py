# evaluate.py: This file probably handles model evaluation. It includes functions to assess
# model performance on validation.

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


def evaluate(model, dataset_loader, device,criterion):
    # implement the evaluation function here
    model.eval()
    total_loss = 0.0
    dice_score = 0.0
    for batch_idx, (images,masks) in enumerate(dataset_loader):
        images, masks = images.to(device),masks.to(device)
        
        outputs = model(images)
        pred_masks = F.sigmoid(outputs)
        
        loss = criterion(pred_masks,masks)

        total_loss += loss.item()
        dice_score += utils.dice_score(pred_masks,masks).item()

    return total_loss / len(dataset_loader) , dice_score / len(dataset_loader) # loss & score