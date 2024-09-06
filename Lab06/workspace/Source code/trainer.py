import torch
import argparse
import os
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from tqdm.auto import tqdm
from dataloader import create_dataloaders
import torchvision.transforms as transforms
from model.ConditionedUNet import ClassConditionedUnet
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from evaluator import evaluation_model
import csv
import os
from torch.cuda import amp

def format_number(num):
    if num >= 1000000000:
        return f"{num / 1000_000_000:.1f}B"
    elif num >= 1000000:
        return f"{num / 1000000:.1f}M"
    elif num >= 1000:
        return f"{num / 1000:.1f}K"
    else:
        return str(num)


def record_training_result(epoch, lr,loss,root, file_name='training_results.csv'):
    # Ensure the root directory exists
    os.makedirs(root, exist_ok=True)

    # Create the full file path
    file_path = os.path.join(root, file_name)
    
    # Check if the file already exists
    file_exists = os.path.isfile(file_path)

    # Define the header
    header = ['epoch', 'lr', 'loss']

    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file doesn't exist
        if not file_exists:
            writer.writerow(header)

        # Write the training result
        writer.writerow([epoch, lr, loss])


class TrainDDPM:

    def __init__(self,args,train_epochs):
        self.args = args
        print(self.args.class_emb_size)
        self.model = ClassConditionedUnet(num_classes=24,class_emb_size=self.args.class_emb_size).to(self.args.device)
        self.train_loader, self.val_loader = create_dataloaders('iclevr', 'train.json', 'objects.json',
                                    batch_size=args.batch_size,
                                    partial=args.partial,
                                    num_workers=args.num_workers,
                                    transform=transform,
                                    val_split=val_split)



        self.train_epochs = train_epochs

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr) 
        self.scheduler =  get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps= len(self.train_loader) * 500,
        )
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
        self.enable_amp = True if args.device.count("cuda") >= 1 else False
        self.scaler = amp.GradScaler(enabled=self.enable_amp)
        print(self.enable_amp)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        # self.accelerator = Accelerator()

        # self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(self.model,self.optimizer,self.train_loader,self.scheduler)
        

        print(f"number of params: {format_number(pytorch_total_params)}")



    def tqdm_bar(self, mode, pbar, loss, avg_loss,lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.0e}" , refresh=False)
        pbar.set_postfix(loss=float(loss),avg_loss=float(avg_loss))
        pbar.refresh()

    def tqdm_bar_acc(self, mode, acc):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, acc:{acc}" , refresh=False)
        pbar.refresh()

    def save_images(self,images, name):
        grid = torchvision.utils.make_grid(images)
        save_image(grid, fp = "./"+name+".png")
    def training_one_epoch(self,epoch):
        self.current_epoch = epoch
        criterion = nn.MSELoss()
        total_loss = 0
        t = 0
        self.model.train()
        for (image,label) in (pbar := tqdm(self.train_loader, ncols=120)):
            
            image = image.to(self.args.device,dtype=torch.float32)
            label = label.squeeze()
            label = label.to(self.args.device,dtype=torch.float32)
            noise = torch.randn_like(image)
            # print(label.shape)
            timesteps = torch.randint(0, 999, (image.shape[0],)).long().to(self.args.device)
            noisy_image = self.noise_scheduler.add_noise(image, noise, timesteps)

            # Get the model prediction
            # print(noisy_image.shape)
            # print(timesteps.shape)
            # print(label.shape)
            with amp.autocast(enabled=self.enable_amp):
                pred = self.model(noisy_image, timesteps, label) # Note that we pass in the labels y

                # Calculate the loss
                loss = criterion(pred, noise) # How close is the output to the noise
            
            self.scaler.scale(loss).backward()
            
            total_loss += loss
            # Backprop and update the params:
            # self.accelerator.backward(loss)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad()
            self.scheduler.step()
            self.optimizer.step()
            
            
            t += 1
            avg_loss = total_loss / t

            # Store the loss for later
            self.tqdm_bar('train', pbar, loss=loss.detach().cpu(),avg_loss=avg_loss, lr=self.scheduler.get_last_lr()[0])

        return avg_loss.item(),self.scheduler.get_last_lr()[0]
    def valid_one_epoch(self,epoch):
        transform=transforms.Compose([
            transforms.Normalize((0, 0, 0), (2, 2, 2)),
            transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
        ])


        # sampling
        with torch.inference_mode():
            for (image,label) in (pbar := tqdm(self.valid_loader, ncols=120)):
                image = image.to(self.args.device,dtype=torch.float32)
                batch_size = image.shape[0]
                
                x = torch.rand(batch_size,3,64,64).to(self.args.device)
                if (batch_size != 1):
                    label = label.squeeze()
                label = label.to(self.args.device,dtype=torch.float32)
                
                for t in (self.noise_scheduler.timesteps):
                    with torch.no_grad():
                        residual = self.model(x,t,label)

                    x = self.noise_scheduler.step(residual,t,x).prev_sample
                
                evaluate = evaluation_model()
                ret = evaluate.eval(x, label)
                print("ACC:",ret)
                img = transform(x)
                self.save_images(img, name="test")
                self.save_images(image, name="ground_truth")




    def save_model(self,epoch,root,suffix=None):
        os.makedirs(root,exist_ok=True)
        if (suffix != None):
            torch.save(self.model.state_dict(), os.path.join(root, f"ddpm_{epoch}_{suffix}.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(root, f"ddpm_{epoch}.pth"))
    
    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDPM")
    
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')  
    parser.add_argument('--save_per_epoch', type=int, default=10, help='Save CKPT per ** epochs(defcault: 1)')

    parser.add_argument('--class_emb_size', type=int, default=512, help='embedding class size')

    parser.add_argument('--ckpt_path', type=str,default="")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu_ids', type=str, default="0", help='gpuid')

    

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    val_split = 0
    if args.test:
        val_split = 0.01
    
    root = f"ddpm_ckpt_{args.class_emb_size}" # try 128 (avoid bottleneck) 512 1024

    ddpmTrainer = TrainDDPM(args,args.epochs)

    if (args.ckpt_path != ""):
        ddpmTrainer.load_model(args.ckpt_path)
    if args.test:
        ddpmTrainer.valid_one_epoch(-1)
    else:
        min_loss = 1e10
        for epoch in range(args.start_epoch,args.epochs+1):
            loss,lr = ddpmTrainer.training_one_epoch(epoch)
            record_training_result(epoch, lr,loss,root)
            #ddpmTrainer.valid_one_epoch(epoch)
            if (loss < min_loss):
                min_loss = loss
                ddpmTrainer.save_model(epoch,root,suffix=f"loss={loss:.3f}")
            elif (epoch  % args.save_per_epoch == 0):
                ddpmTrainer.save_model(epoch,root)

