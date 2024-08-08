# Implement your UNet model here
import math

import torch.utils.checkpoint as checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop


class UNet(nn.Module):

#  >contracting path<                              >expansive path<
    #                                             cc1
    #  [left_1] -c1-----------(c&c)-------------------->[right_1] e1
    #        |                                           | 
    #      (ds1)                                       (us1)
    #        |                               cc2         | 
    #       [left_2] -c2-------(c&c)------------->[right_2] e2
    #             |                                |
    #           (ds2)                            (us2)
    #             |                       cc3      |
    #            [left_3] -c3--(c&c)------->[right_3] e3
    #                  |                     |
    #                (ds3)                 (us3)
    #                  |       c4     cc4    |
    #                 [left_4]-(c&c)->[right_4] e4
    #                       |          |
    #                     (ds4)      (us4)
    #                       |          |
    #                      [bottom layer]
    #                          bottom
    # (c&c) copy and crop 
    def __init__(self,input_channel,output_channel):
        super(UNet, self).__init__()
        
        self.left_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=64,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.left_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.left_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.left_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.right_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.right_3 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.right_2 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.right_1 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=output_channel,kernel_size=(1,1)),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.ds1 = nn.MaxPool2d(2)
        self.ds2 = nn.MaxPool2d(2)
        self.ds3 = nn.MaxPool2d(2)
        self.ds4 = nn.MaxPool2d(2)
        
        self.us4 = nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=(2,2),stride=(2,2))
        self.us3 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(2,2),stride=(2,2))
        self.us2 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(2,2),stride=(2,2))
        self.us1 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(2,2),stride=(2,2))

    def crop(self, input_feature, x):
        (_ , _ , H , W)  = x.shape
        crop_obj = CenterCrop([H, W])
        output_feature = crop_obj(input_feature)

        return output_feature
        
    def forward(self,x):
        self.c1 = self.left_1(x)

        self.c2 = self.left_2(self.ds1(self.c1))

        self.c3 = self.left_3(self.ds2(self.c2))

        self.c4 = self.left_4(self.ds3(self.c3))

        self.bottom_value = self.bottom(self.ds4(self.c4))


        x = self.us4(self.bottom_value)
        #print("-",x.shape)
        self.cc4 = torch.cat(
            [
                self.crop(self.c4,x),
                x
            ],dim=1)
        #print("->",self.cc4.shape)

        
        self.e4 = self.right_4(self.cc4)

        x = self.us3(self.e4)
        #print("-",x.shape)
        self.cc3 = torch.cat(
            [
                self.crop(self.c3,x),
                x
            ],dim=1)
        #print("->",self.cc3.shape)
        
        self.e3 = self.right_3(self.cc3)

        x = self.us2(self.e3)
        #print("-",x.shape)
        self.cc2 = torch.cat(
            [
                self.crop(self.c2,x),
                x
            ],dim=1)
        #print("->",self.cc2.shape)
        
        self.e2 = self.right_2(self.cc2)


        x = self.us1(self.e2)
        #print("-",x.shape)
        self.cc1 = torch.cat(
            [
                self.crop(self.c1,x),
                x
            ],dim=1)
        #print("->",self.cc1.shape)
        
        self.e1 = self.right_1(self.cc1) # output channel 1
        return self.e1

#  >contracting path<                              >expansive path<
    #                                             cc1
    #  [left_1] -c1-----------(c&c)-------------------->[right_1] e1
    #        |                                           | 
    #      (ds1)                                       (us1)
    #        |                               cc2         | 
    #       [left_2] -c2-------(c&c)------------->[right_2] e2
    #             |                                |
    #           (ds2)                            (us2)
    #             |                       cc3      |
    #            [left_3] -c3--(c&c)------->[right_3] e3
    #                  |                     |
    #                (ds3)                 (us3)
    #                  |       c4     cc4    |
    #                 [left_4]-(c&c)->[right_4] e4
    #                       |          |
    #                     (ds4)      (us4)
    #                       |          |
    #                      [bottom layer]
    #                          bottom
    # (c&c) copy and crop 
