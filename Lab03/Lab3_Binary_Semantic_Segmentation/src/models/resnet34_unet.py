# Implement your ResNet34_UNet model here
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        out = self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return x * self.sigmoid(x_out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x





class buildingblock(nn.Module):
    #    x
    # [weight]-\
    #    |     |
    #  (relu)  |
    #    |     |
    # [weight] | (identity)
    #    |     |
    #  ( + )---/
    #    |
    #  (relu)
    #    
    # [3x3 64] x 2
    # [3x3 64]
    #
    # [3x3 128]
    # [3x3 128]
    # 
    # [3x3 256] 
    # [3x3 256]
    #
    # [3x3 512] 
    # [3x3 512]

    def __init__(self,input_channel,output_channel,is_first_block=False):
        super().__init__()
        if (is_first_block):
            self.conv_1 = nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=(3,3),stride=2,padding=1,bias=False)
        else:
            self.conv_1 = nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=(3,3),padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_channels=output_channel,out_channels=output_channel,kernel_size=(3,3),padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        
        self.is_first_block = is_first_block
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.diff_sz_conv = nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=(1,1),stride=2,bias=False)

    def forward(self,x):
        self.identity = x

        if (self.input_channel != self.output_channel):
            self.identity = self.diff_sz_conv(self.identity)

        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.bn2(x)

        
        #print(x.shape,self.identity.shape,self.input_channel,self.output_channel)
        x += self.identity
        x = self.relu(x)
        return x



class ResNet34_UNet(nn.Module):
    def __init__(self, input_channel=3,output_channel=1):
        super(ResNet34_UNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=input_channel,out_channels=64,kernel_size=(7,7),stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1)


        #x3
        self.conv_2 = nn.Sequential(
            buildingblock(64,64),
            buildingblock(64,64),
            buildingblock(64,64)
        )


        #x4
        self.conv_3 = nn.Sequential(
            buildingblock(64,128,is_first_block=True),
            buildingblock(128,128),
            buildingblock(128,128),
            buildingblock(128,128)
        )

        #x6
        self.conv_4 = nn.Sequential(
            buildingblock(128,256,is_first_block=True),
            buildingblock(256,256),
            buildingblock(256,256),
            buildingblock(256,256),
            buildingblock(256,256),
            buildingblock(256,256)
        )

        #x3
        self.conv_5 = nn.Sequential(
            buildingblock(256,512,is_first_block=True),
            buildingblock(512,512),
            buildingblock(512,512)
        )


        self.us4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=768,out_channels=32,kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32)
        )
        self.us3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=288,out_channels=32 ,kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32)
        )
        self.us2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=160,out_channels=32 ,kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32)
        )
        self.us1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=96,out_channels=32  ,kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32)
        )
        self.us0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32,out_channels=32   ,kernel_size=(2,2),stride=(2,2)), # 多做一次 decode
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32)
        )

        self.right_0 = nn.Sequential( #output
            nn.Conv2d(in_channels=32,out_channels=output_channel,kernel_size=(1,1)),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            
        )

        self.bottom = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(3,3),padding=1)





    def forward(self,x):
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)



        self.c1 = self.conv_2(x)

        self.c2 = self.conv_3(self.c1)

        self.c3 = self.conv_4(self.c2)

        self.c4 = self.conv_5(self.c3)



        
        x = self.bottom(self.c4)
        #print("-",x.shape)
        self.cc4 = torch.cat(
            [
                x,
                self.c4
            ],dim=1)
        #print("->",self.cc4.shape)

        
        self.e4 = self.us4(self.cc4)

        x = self.e4
        #print("-",x.shape)
        self.cc3 = torch.cat(
            [
                x,
                self.c3
            ],dim=1)
        #print("->",self.cc3.shape)
        
        self.e3 = self.us3(self.cc3)

        x = self.e3
        #print("-",x.shape)
        self.cc2 = torch.cat(
            [
                x,
                self.c2
            ],dim=1)
        #print("->",self.cc2.shape)
        
        self.e2 = self.us2(self.cc2)


        x = self.e2
        #print("-",x.shape)
        self.cc1 = torch.cat(
            [
                x,
                self.c1
            ],dim=1)
        #print("->",self.cc1.shape)
        
        self.e1 = self.us1(self.cc1) 

        self.e0 = self.right_0(self.us0(self.e1))# output channel 1

    
        return self.e0


#          >encode<                                     >decode<
    #                                             cc1
    #  [conv_2] -c1------------------------------------>[right_1] e1  --(us1)--> [ ]
    #        |                                           | 
    #        |                                         (us2)
    #        |                               cc2         | 
    #       [conv_3] -c2------------------------->[right_2] e2
    #             |                                |
    #             |                              (us3)
    #             |                       cc3      |
    #            [conv_4] -c3-------------->[right_3] e3
    #                  |                     |
    #                  |                   (us4)
    #                  |       c4     cc4    |
    #                 [conv_5]------------->[right_4] e4
    #                   \-------(bottom)-----/
