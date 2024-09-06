import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

class ClassConditionedUnet(nn.Module):
  def __init__(self, num_classes=10, class_emb_size=4):
    super().__init__()
    
    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Linear(num_classes, class_emb_size)
    self.class_emb_size = class_emb_size
    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=64,           # the target image resolution
        in_channels=3+class_emb_size, # Additional input channels for class cond.
        out_channels=3,           # the number of output channels
        layers_per_block=2,       # how many ResNet layers to use per UNet block
        #block_out_channels=(32, 64, 128,256,512),  # 
        # block_out_channels=(128,128,256,256,512),
        block_out_channels=(128,128,256,256,512,512),
        down_block_types=( 
            "DownBlock2D",        # a regular ResNet downsampling block
            "DownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",        # a regular ResNet downsampling block
            "DownBlock2D",        # a regular ResNet downsampling block
            "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D", 
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",          # a regular ResNet upsampling block
            "UpBlock2D",          # a regular ResNet upsampling block
            "UpBlock2D", 
            "UpBlock2D",          # a regular ResNet upsampling block
          ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape
    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels) # Map to embedding dimension
    class_cond = class_cond.view(bs, self.class_emb_size, 1, 1).expand(bs, self.class_emb_size, w, h)
    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) 
    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample

