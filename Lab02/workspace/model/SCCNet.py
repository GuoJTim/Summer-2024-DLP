# implement SCCNet model
# ▪ Implement the SCCNet architecture, read the paper before doing the
#   implementation.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# reference paper: https://ieeexplore.ieee.org/document/8716937
class SquareLayer(nn.Module):
    def __init__(self):
        super(SquareLayer, self).__init__()
    
    def forward(self, x):
        return x ** 2

class LogLayer(nn.Module):
    def __init__(self):
        super(LogLayer, self).__init__()
    
    def forward(self, x):
        return torch.log(x + 1)

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=438, Nu=0, C=22, Nc=20, Nt=0, dropoutRate=0.5):

        super(SCCNet, self).__init__()
        # Zero-padding, batch normalization were applied to first and second conv
        # l2 regularization with coefficient 0.0001

        self.first_conv = nn.Conv2d(in_channels=1,out_channels=Nu,kernel_size=(C,Nt),padding=(0,0))
        first_conv_sz = self.get_conv_size(C,timeSample,in_channels=1,out_channels=Nu,kernel_size=(C,Nt),padding=(0,0))
        
        self.first_batch_norm = nn.BatchNorm2d(Nu)

        self.second_conv = nn.Conv2d(in_channels=Nu,out_channels=Nc,kernel_size=(1,12),padding=(0,6))

        second_conv_sz = self.get_conv_size(first_conv_sz[1],first_conv_sz[2],in_channels=Nu,out_channels=Nc,kernel_size=(1,12),padding=(0,6))
        
        self.second_batch_norm = nn.BatchNorm2d(Nc)
        self.second_square = SquareLayer()

        self.dropout = nn.Dropout(dropoutRate)
        self.pooling = nn.AvgPool2d(kernel_size=(1,62),stride=(1,12))
        pool_zx = self.get_pool_size(second_conv_sz[1],second_conv_sz[2],kernel_size=(1,62),stride=(1,12))
        
        self.log_activation = LogLayer()


        self.fc = nn.Linear(self.get_size(second_conv_sz[0],pool_zx[1]),numClasses)
        

    def forward(self, x):
        # 288, 22, 438
        # 288 = 6 * 12 * 4   (total trails /times)
        #       6  總共 run 次數
        #       每一 run 12 * 4 表示可能的classes，每個 classes 佔 12 trails 
        #
        # 22 -> EEG Channel (Twenty-two Ag/AgCl electrodes)
        #
        # 438 每個電極的特徵點

        # Each session is comprised of 6 runs separated by
        # short breaks. One run consists of 48 trials (12 for each of the four possible
        # classes), yielding a total of 288 trials per session.
        # 
        # input batch_size, EEG CHANNEL(22)
        # print("INPUT SHAPE:",x.shape)
        x = x.unsqueeze(1) # B x 1 x 22 x438
        # print("UNSQUEEZE SHAPE:",x.shape)
        x = self.first_conv(x) # 22 x 1 x438
        x = self.first_batch_norm(x) 
        
        # permute
        # print("FIRST CONV SHAPE:",x.shape)
        #x = x.permute(0, 2 ,1 ,3)
        # print("PERMUTE SHAPE:",x.shape)

        x = self.second_conv(x) # 20 x 1 x 439
        x = self.second_batch_norm(x)
        x = self.second_square(x)
        # print("SECOND CONV SHAPE:",x.shape)
        x = self.dropout(x)
        x = self.pooling(x) # 1x 32
        # print("AVG POOL SHAPE:",x.shape)


        x = self.log_activation(x)
        #print("LOG ACT:",x.shape)
        
        x = torch.flatten(x,1)
        #x = x.view(x.size(0), -1)
        # print("FLATTEN:",x.shape)


        #print("FC SHAPE:",x.shape)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    def get_conv_size(self,input_h,input_w,in_channels,out_channels,kernel_size,padding=(0,0),stride=(1,1)):
        kernel_size = list(kernel_size)
        padding = list(padding)
        stride = list(stride)
        return [out_channels,(input_h - kernel_size[0] + 2 * padding[0]) // stride[0] + 1 , (input_w - kernel_size[1] + 2 * padding[1]) // stride[1] + 1]
    
    def get_pool_size(self,input_h,input_w,kernel_size,padding=(0,0),stride=(1,1)):
        kernel_size = list(kernel_size)
        padding = list(padding)
        stride = list(stride)
        return  [math.floor( ( input_h - kernel_size[0] + 2 * padding[0] ) / stride[0]) + 1,
                 math.floor( ( input_w - kernel_size[1] + 2 * padding[1] ) / stride[1]) + 1]
    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        return C*N