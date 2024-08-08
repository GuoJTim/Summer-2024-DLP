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
from evaluate import evaluate
import tqdm
import time
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args,model,criterion,optimizer,train_loader):
    global device

    model.train()
    total_loss = 0.0
    for batch_idx, (images,masks) in enumerate(train_loader):
        images, masks = images.to(device, dtype=torch.float32),masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        outputs = model(images)
        pred_masks = F.sigmoid(outputs)
        # print(pred_masks.shape,masks.shape)
        # pred_masks = pred_masks.squeeze(1)
        # masks = masks.squeeze(1)
        # print(pred_masks.shape,masks.shape)
        loss = criterion(pred_masks,masks)
        dice_score = utils.dice_score(pred_masks,masks).item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print("finish ",batch_idx," loss:",loss.item()," ds:",dice_score)
    return total_loss / len(train_loader)

def save_model_weight(model, optimizer, epoch, last_loss,last_score, file_path):
    # Create directory if it does not exist
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Save the model and optimizer state
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'loss': last_loss,
        'score': last_score
    }
    torch.save(state, file_path)
    print(f"Checkpoint saved to {file_path}")

def load_model_weight(model, optimizer, file_path):
    # Load the model state
    if os.path.isfile(file_path):
        state = torch.load(file_path)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
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


def training_epoch(args):
    global device
    train_ds = load_dataset(args.data_path ,"train")
    valid_ds = load_dataset(args.data_path ,"valid")
    train_loader = torch.utils.data.DataLoader(dataset=train_ds,batch_size=args.batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_ds,batch_size=args.batch_size,shuffle=False)

    model_name = ""
    if (args.model_name == "unet"):
        model = UNet(input_channel=3,output_channel=1).to(device) # binary semantic segmentation,
        model_name = "unet"
    else:
        model = ResNet34_UNet(input_channel=3,output_channel=1).to(device) 
        model_name = "resnet_unet"
        pass

    criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.SGD(model.parameters(),lr=args.learning_rate,momentum=0.99)
    optimizer = optim.AdamW(model.parameters(),lr=args.learning_rate)

    last_epoch,last_loss = 0,0
    if (args.model is not None):
        last_epoch,last_loss = load_model_weight(model,optimizer,args.model)
    
    torch.cuda.empty_cache()
    

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Start training : {current_time}")
    for epoch in range(last_epoch,args.epochs):
        train_loss = train(args,model,criterion,optimizer,train_loader)
        valid_loss,last_score = evaluate(model,valid_loader,device,criterion)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{current_time}| [{epoch+1}/{args.epochs}] Train Loss: {train_loss} Valid Loss: {valid_loss} Dice Score: {last_score}")
        save_model_weight(model,optimizer,epoch+1,train_loss,last_score,f"../saved_models/{model_name}_{epoch+1}.pth")





def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', default=None, help='path to the stored model weight')
    parser.add_argument('--model_name', default='resnet34_unet', help='unet or resnet34_unet')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0005, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    training_epoch(args)