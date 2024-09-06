import time, torchvision, argparse, sys, os
import torch, random
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim

from datasets.datasets_pairs import my_dataset, my_dataset_eval,my_dataset_wTxt,FusionDataset
from datasets.reflect_dataset_for_fusion import CEILDataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from utils.UTILS import compute_psnr,MixUp_AUG,rand_bbox
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from loss.perceptual import LossNetwork
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loss.contrastive_loss import HCRLoss
from networks.network_RefDet import RefDet,RefDetDual
trans_eval = transforms.Compose(
    [
        transforms.ToTensor()
    ])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:',device)


parser = argparse.ArgumentParser()
parser.add_argument('--eval_in_path_nature20', type=str, default='/app/nature20/blended/')
parser.add_argument('--eval_gt_path_nature20', type=str, default='/app/nature20/transmission_layer/')

parser.add_argument('--eval_in_path_real20', type=str, default='/app/real20/blended/')
parser.add_argument('--eval_gt_path_real20', type=str, default='/app/real20/transmission_layer/')

parser.add_argument('--eval_in_path_wild55', type=str, default='/app/wild55/blended/')
parser.add_argument('--eval_gt_path_wild55', type=str, default='/app/wild55/transmission_layer/')

parser.add_argument('--eval_in_path_soild200', type=str, default='/app/solid200/blended/')
parser.add_argument('--eval_gt_path_soild200', type=str, default='/app/solid200/transmission_layer/')

parser.add_argument('--eval_in_path_postcard199', type=str, default='/app/postcard199/blended/')
parser.add_argument('--eval_gt_path_postcard199', type=str, default='/app/postcard199/transmission_layer/')


#  --in_norm   --pyramid
args = parser.parse_args()

def get_eval_data(val_in_path=args.eval_in_path_nature20, val_gt_path=args.eval_gt_path_nature20
                  , trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label=val_gt_path, transform=trans_eval, fix_sample=500)

    # eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_data)
    # eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4,sampler=eval_sampler)

    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader



eval_loader_nature20 = get_eval_data(val_in_path=args.eval_in_path_nature20, val_gt_path=args.eval_gt_path_nature20)
eval_loader_real20 = get_eval_data(val_in_path=args.eval_in_path_real20, val_gt_path=args.eval_gt_path_real20)
eval_loader_wild55 = get_eval_data(val_in_path=args.eval_in_path_wild55, val_gt_path=args.eval_gt_path_wild55)
eval_loader_postcard199 = get_eval_data(val_in_path=args.eval_in_path_postcard199, val_gt_path=args.eval_gt_path_postcard199)
eval_loader_soild200 = get_eval_data(val_in_path=args.eval_in_path_soild200, val_gt_path=args.eval_gt_path_soild200)

print(len(eval_loader_soild200))