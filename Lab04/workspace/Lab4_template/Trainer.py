import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio


import matplotlib.pyplot as plt
from math import log10
import csv
import os

def record_training_result(epoch, beta, avg_loss, avg_psnr, result_type, tfr, tf, lr, root, file_name='training_results.csv'):
    # Ensure the root directory exists
    os.makedirs(root, exist_ok=True)

    # Create the full file path
    file_path = os.path.join(root, file_name)
    
    # Check if the file already exists
    file_exists = os.path.isfile(file_path)

    # Define the header
    header = ['epoch', 'beta', 'avg_loss', 'avg_psnr', 'type','teacher forcing ratio','teacher forcing','learning_rate']

    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file doesn't exist
        if not file_exists:
            writer.writerow(header)

        # Write the training result
        writer.writerow([epoch, beta, avg_loss, avg_psnr, result_type,tfr,tf,lr])



seed = 48763

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD

class tf_detector():
    
    def __init__(self, args, patience=5, threshold=0.01):
        self.args = args
        self.valid_psnr_record = []
        self.valid_loss_record = []
        self.next_ratio = 1
        self.patience = patience
        self.threshold = threshold

    def add_valid_record(self, psnr, loss):
        self.valid_psnr_record.append(psnr)
        self.valid_loss_record.append(loss)

    def check_convergence(self, records):
        if len(records) < self.patience + 1:
            return False
        
        recent_records = records[-self.patience:]
        deltas = [abs(recent_records[i] - recent_records[i-1]) for i in range(1, len(recent_records))]
        
        if all(delta < self.threshold for delta in deltas):
            return True
        return False

    # update the teacher forcing ratio to 1 if convergence in validation metrics is detected
    def update(self, current_tfr):
        if current_tfr > 0:
            self.next_ratio = current_tfr  # keep the ratio
            return
        
        if (self.check_convergence(self.valid_psnr_record) and 
            self.check_convergence(self.valid_loss_record)):
            self.next_ratio = 1  # Set the ratio to 1 if convergence detected in validation metrics
            self.valid_psnr_record = []
            self.valid_loss_record = []
            # clear array waiting new inputs (which after teacher forcing ended up, then recalc the ratio)

    def get_TF_ratio(self):
        return self.next_ratio

class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.args = args
        self.iter = current_epoch
        
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        if self.kl_anneal_type == "NONE":
            self.beta = 1
        else:
            self.beta = 0
        if (self.iter != 0):
            self.iter -= 1 # shift
            self.update()
    def update(self):
        # TODO
        # 更新 beta
    #parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    #parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    #parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
        #raise self.kl_anneal_type in ["Cyclical","Monotonic"]
        self.iter += 1 # update
        if self.kl_anneal_type == "Cyclical":
            self.beta = self.frange_cycle_linear(n_iter=self.args.num_epoch,start=0.0,stop=1.0,n_cycle=self.kl_anneal_cycle,ratio=self.kl_anneal_ratio)[self.iter]  # current beta
        elif self.kl_anneal_type == "Monotonic":
            self.beta = self.frange_cycle_linear(n_iter=self.args.num_epoch,start=0.0,stop=1.0,n_cycle=1,ratio=0.25)[self.iter]  # current beta
        elif self.kl_anneal_type == "NONE":
            self.beta = 1
    
    def get_beta(self):
        return self.beta


    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        # IN PAPER
        # \beta_t = f(t) , \tau <= R
        #         = 1    , \tau > R 
        # 
        # \tau = mod(t-1, ceil(T/M) ) / (T / M)
        #       
        #  T : total number of training iteratinos
        #  M,R hyper parameter
        # 
        # update beta
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2,5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label): 
#        self.frame_transformation = RGB_Encoder(3, args.F_dim)
#        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
#        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
#        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
#        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        # [B, seq, RGB, W, H]

        # for each batch , there may have a sequence {x1,x2, ... ,xk}
        # to generate pred_x 
        pass
        
       



    
    def training_stage(self):
        if (self.args.TF_detector):
            self.tf_detector = tf_detector(args,args.detector_patience,args.detector_threshold)


        max_psnr = -1
        min_loss = 1e5
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            t = 0
            total_loss = 0
            total_psnr = 0
            for (img, label) in (pbar := tqdm(train_loader, ncols=180)) : #
                # [B, seq, RGB, W, H]
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                
                avg_psnr,loss,error = self.training_one_step(img, label, adapt_TeacherForcing)
                if (error):
                    # gradient explosion
                    continue
                
                loss_value = loss.detach().cpu()

                total_loss += loss_value
                total_psnr += avg_psnr
                t += 1 
                avg_loss = total_loss / t
                ent_psnr = total_psnr / t
 
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {:.3f}'.format(self.tfr, beta), pbar, loss_value, lr=self.scheduler.get_last_lr()[0],psnr=avg_psnr,avg_loss=avg_loss,avg_psnr=ent_psnr)
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.3f}'.format(self.tfr, beta), pbar, loss_value, lr=self.scheduler.get_last_lr()[0],psnr=avg_psnr,avg_loss=avg_loss,avg_psnr=ent_psnr)
            
            record_training_result(epoch=self.current_epoch,beta=self.kl_annealing.get_beta(),avg_loss=float(total_loss / t),avg_psnr=float(total_psnr / t),root=self.args.save_root,result_type="train",tfr=self.tfr,tf=adapt_TeacherForcing,lr=self.scheduler.get_last_lr()[0])
            ret_psnr,ret_loss = self.eval()
            record_training_result(epoch=self.current_epoch,beta=self.kl_annealing.get_beta(),avg_loss=float(ret_loss),avg_psnr=float(ret_psnr),root=self.args.save_root,result_type="valid",tfr=self.tfr,tf=adapt_TeacherForcing,lr=self.scheduler.get_last_lr()[0])
            

            epoch_saved = False

            if (self.args.auto_save):
                # lowest eval loss
                # highest psnr value
                if (ret_psnr >= max_psnr and ret_loss <= min_loss):
                    max_psnr = ret_psnr
                    min_loss = ret_loss
                    self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}_psnr={max_psnr:.3f}_loss={ret_loss:.3f}.ckpt"))
                    epoch_saved = True
                elif (ret_loss <= min_loss):
                    min_loss = ret_loss
                    self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}_loss={ret_loss:.3f}.ckpt"))
                    epoch_saved = True
                elif (ret_psnr >= max_psnr):
                    max_psnr = ret_psnr
                    self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}_psnr={max_psnr:.3f}.ckpt"))
                    epoch_saved = True

            if self.current_epoch % self.args.per_save == 0 and not epoch_saved:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            self.current_epoch += 1
            if (not self.args.continue_training):
                self.scheduler.step()
            
            if (self.args.TF_detector):
                self.tf_detector.add_valid_record(psnr=ret_psnr,loss=ret_loss)
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=180)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            ret_psnr,loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0],psnr=ret_psnr,avg_loss=-1,avg_psnr=-1)
        
        return ret_psnr,loss.detach().cpu()
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        #
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        
        psnr_values = []
        #self.optim.zero_grad()
        self.frame_transformation.zero_grad() # 
        self.label_transformation.zero_grad() 
        self.Gaussian_Predictor.zero_grad()
        self.Decoder_Fusion.zero_grad()
        self.Generator.zero_grad()
        
        
        #self.forward(img,label)
        # [B, seq, RGB, W, H]
        decoded_frame_list = [img[0]]
        label_list = [label[0]]
        seq_len = img.shape[0]
        batch_size = img.shape[1]
        kld = 0
        mse = 0

        for seq_id in range(1,seq_len):
            # First frame is the past frame which is provided to predict the consecutive k-1 future frames
            x_predictor = img[seq_id] # current  
            
            if (adapt_TeacherForcing):
                x_prev = img[seq_id-1]
            else:
                x_prev = decoded_frame_list[seq_id-1]
                
            p = label[seq_id]
            label_list.append(p)

            
            x_generator_frame_encoder = self.frame_transformation(x_prev)
            x_predictor_frame_encoder = self.frame_transformation(x_predictor)
            
            if (torch.isnan(x_predictor_frame_encoder).any() or torch.isnan(x_generator_frame_encoder).any()):
                print("ERROR")
                return 0,0,True 
            
            p_pose_encoder = self.label_transformation(p)

            z, mu, logvar = self.Gaussian_Predictor(
                x_predictor_frame_encoder,                
                p_pose_encoder
            )
            
            
            decoder_fusion_output = self.Decoder_Fusion(
                x_generator_frame_encoder,
                p_pose_encoder,
                z
            )
            gen_images = self.Generator(decoder_fusion_output)

            decoded_frame_list.append(gen_images)
            psnr = Generate_PSNR(gen_images,img[seq_id])
            psnr_values.append(psnr.item())  # Convert tensor to scalar


            
            if (torch.isnan(gen_images).any() or 
                torch.isnan(x_predictor_frame_encoder).any() or
                torch.isnan(x_generator_frame_encoder).any() or
                torch.isnan(p_pose_encoder).any() or 
                torch.isnan(z).any() or
                torch.isnan(decoder_fusion_output).any() or
                torch.isnan(gen_images).any() or
                torch.isnan(self.mse_criterion(gen_images,img[seq_id])).any() ):

                print(seq_id)
                print(torch.isnan(x_predictor).any(),
                        torch.isnan(x_prev).any(),
                        torch.isnan(x_predictor_frame_encoder).any(),
                        torch.isnan(x_generator_frame_encoder).any(),
                        torch.isnan(p_pose_encoder).any(),
                        torch.isnan(z).any(),
                        torch.isnan(decoder_fusion_output).any(),
                        torch.isnan(gen_images).any())
                return 0,0,True

            

            kld_v =kl_criterion(mu=mu,logvar=logvar,batch_size=batch_size) 
            mse_v = self.mse_criterion(gen_images,img[seq_id])
            kld += kld_v
            # print(gen_images.shape,gen_images.dtype)
            # print(img[seq_id].shape,img[seq_id].dtype)
            mse += mse_v 
        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        label_frame = stack(label_list).permute(1, 0, 2, 3, 4)
        
        img = img.permute(1, 0, 2, 3, 4)
        # self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'pred_seq.gif'))
        

        beta = self.kl_annealing.get_beta()
        # print(mse,kld,kld/seq_len)
        avg_psnr = sum(psnr_values) / len(psnr_values)
        loss = mse + kld * beta 

        loss.backward()
        self.optimizer_step()
        # print(kld,mse) 

        
        return avg_psnr,loss,False
       
    def val_one_step(self, img, label):
        # for the inference or same as the training ??
        # TODO
        #
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        #self.forward(img,label)
        # [B, seq, RGB, W, H]
        psnr_values = []

        decoded_frame_list = [img[0]]
        label_list = [label[0]]
        seq_len = img.shape[0]
        batch_size = img.shape[1]
        kld = 0
        mse = 0
        num_frames = seq_len-1

        for seq_id in range(1,seq_len):
            # First frame is the past frame which is provided to predict the consecutive k-1 future frames
            x_prev = decoded_frame_list[seq_id-1]
            x_ground_truth = img[seq_id] # current  
            
            p = label[seq_id]
            label_list.append(p)

            
            x_generator_frame_encoder = self.frame_transformation(x_prev)
            p_pose_encoder = self.label_transformation(p)

            z = torch.randn(batch_size,self.args.N_dim,self.args.frame_H,self.args.frame_W).to(self.args.device)
            
            
            decoder_fusion_output = self.Decoder_Fusion(
                x_generator_frame_encoder,
                p_pose_encoder,
                z
            )
            gen_images = self.Generator(decoder_fusion_output)
            
            decoded_frame_list.append(gen_images)
            psnr = Generate_PSNR(gen_images,x_ground_truth)
            psnr_values.append(psnr.item())  # Convert tensor to scalar


            mse += self.mse_criterion(gen_images,x_ground_truth) 
        

        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        label_frame = stack(label_list).permute(1, 0, 2, 3, 4)
        
        img = img.permute(1, 0, 2, 3, 4)
        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'valid_seq.gif'))

        # print(mse,kld,kld/seq_len)
        loss = mse 
        
        avg_psnr = sum(psnr_values) / len(psnr_values)
        # print(avg_psnr)
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_frames), psnr_values, label=f'PSNR per frame')
        plt.title('Per Frame Quality (PSNR)')
        plt.xlabel('Frame Index')
        plt.ylabel('PSNR')
        plt.legend(loc='upper right', title=f'Avg_PSNR: {avg_psnr:.2f}')
        plt.grid(True)
        plt.savefig('valid_psnr.png')



        return avg_psnr,loss
        
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def finetune_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='finetune', video_len=self.train_vi_len, \
                                                partial=1.0)
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch >= self.args.tfr_sde:
            self.tfr = max(self.tfr - self.tfr_d_step, 0)
        
        if (self.args.TF_detector):
            self.tf_detector.update(current_tfr=self.tfr)
            self.tfr = self.tf_detector.get_TF_ratio()
        
            
    def tqdm_bar(self, mode, pbar, loss, lr,psnr,avg_loss,avg_psnr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.0e}" , refresh=False)
        pbar.set_postfix(loss=float(loss),psnr=float(psnr), refresh=False,avg_loss=float(avg_loss),avg_psnr=float(avg_psnr))
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            if (not self.args.use_new_lr):
                self.args.lr = checkpoint['lr']
            if (not self.args.reset_teacher_forcing):
                self.tfr = checkpoint['tfr']
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2,5], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=400,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    parser.add_argument('--auto_save',    action='store_true' ,help="auto save best ckpt")
    parser.add_argument('--continue_training',    action='store_true' ,help="continue training dont change the learning rate schedular")
    parser.add_argument('--reset_teacher_forcing',    action='store_true' ,help="dont load the teacher forcing ratio from ckpt")
    parser.add_argument('--use_new_lr',    action='store_true' ,help="dont load the learning rate from ckpt")

    
    parser.add_argument('--TF_detector',    action='store_true' ,help="use the auto teacher forcing ratio strategy")
    parser.add_argument('--detector_patience',    type=int, default=5, help="last N patience")
    parser.add_argument('--detector_threshold',    type=float, default=0.01, help="threshold that control if convergence")
    
    args = parser.parse_args()
    # Construct the command string
    cmd_str = "python " + " ".join([f"--{k} {v}" for k, v in vars(args).items() if v is not None and v is not False])

    # File path for cmd.txt
    cmd_file_path = os.path.join(args.save_root, 'cmd.txt')

    os.makedirs(os.path.dirname(cmd_file_path), exist_ok=True)


    # Write or append the command to cmd.txt
    with open(cmd_file_path, 'a') as f:
        f.write(cmd_str + '\n')
    main(args)
