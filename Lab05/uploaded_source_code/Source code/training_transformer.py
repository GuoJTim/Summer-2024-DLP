import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.args = args
        self.loss = F.cross_entropy
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)


    def tqdm_bar(self, mode, pbar, loss, avg_loss,lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.0e}" , refresh=False)
        pbar.set_postfix(loss=float(loss),avg_loss=float(avg_loss))
        pbar.refresh()
    def train_one_epoch(self,train_loader,cur_epoch):
        self.current_epoch = cur_epoch
        total_loss = 0
        cnt = 0
        self.model.train()
        for (img) in (pbar := tqdm(train_loader, ncols=180)) : #
            cnt += 1
            self.optim.zero_grad()

            img = img.to(self.args.device)
            logits, z_indices, mask  = self.model(img) 
            # [10,1025] | [10,256]


            #loss = self.loss(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            batch_size , _ = z_indices.size()
            log_probs = F.log_softmax(logits, dim=-1)
            
            masked_positions = mask.repeat(batch_size,1)
            masked_log_probs = log_probs[masked_positions, z_indices[masked_positions]]
            
            loss = -masked_log_probs.mean()
            if (masked_log_probs.shape[0] == 0):
                continue 
                #skip
            #loss = self.loss(logits,target)
            total_loss += loss
            avg_loss = total_loss / cnt
            loss.backward()
            self.optim.step()
            self.tqdm_bar('train ', pbar, loss=loss.detach().cpu(),avg_loss=avg_loss, lr=self.scheduler.get_last_lr()[0])
        self.scheduler.step(total_loss)

    def eval_one_epoch(self,valid_loader,cur_epoch):
        self.current_epoch = cur_epoch
        total_loss = 0
        cnt = 0
        #self.model.eval()
        with torch.inference_mode():
            for (img) in (pbar := tqdm(valid_loader, ncols=180)) : #
                cnt += 1
                img = img.to(self.args.device)
                logits, z_indices,mask  = self.model(img)


                #loss = self.loss(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                
                batch_size , _ = z_indices.size()
                log_probs = F.log_softmax(logits, dim=-1)

                masked_positions = mask.repeat(batch_size,1)
                masked_log_probs = log_probs[masked_positions, z_indices[masked_positions]]
                
                loss = -masked_log_probs.mean()
                if (masked_log_probs.shape[0] == 0):
                    continue 
                    #skip

                total_loss += loss
                avg_loss = total_loss / cnt        

                self.tqdm_bar('valid ', pbar, loss=loss.detach().cpu(),avg_loss=avg_loss, lr=self.scheduler.get_last_lr()[0])
            

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.transformer.parameters(), lr=0.0001, betas=(0.9, 0.96), weight_decay=4.5e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-6, patience=2)
        # scheduler = None
        return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum_grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--save_per_epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start_from_epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt_interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=0, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_transformer.train_one_epoch(train_loader,epoch)
        train_transformer.eval_one_epoch(val_loader,epoch)
        if epoch % args.save_per_epoch == 0:
            # save
            torch.save(train_transformer.model.transformer.state_dict(), os.path.join("transformer_checkpoints", f"transformer_epoch_{epoch}.pt"))
        torch.save(train_transformer.model.transformer.state_dict(), os.path.join("transformer_checkpoints", "transformer_current.pt"))
