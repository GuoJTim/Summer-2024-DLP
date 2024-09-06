import torch
import json
import argparse
import os
import torchvision
import numpy as np
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
import csv
from evaluator import evaluation_model

seed = 48763

#random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def record_testing_result(epoch, accuracy, root, class_emb_size,file_name='accuracy_results.csv'):
    # 确保目录存在
    os.makedirs(root, exist_ok=True)

    # 创建文件路径
    file_path = os.path.join(root, file_name)
    
    # 检查文件是否已存在
    file_exists = os.path.isfile(file_path)

    # 定义表头
    header = ['epoch', 'accuracy','class_emb_size']

    # 以追加模式打开文件
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(header)

        # 写入训练结果
        writer.writerow([epoch, accuracy,class_emb_size])


def format_number(num):
    if num >= 1000000000:
        return f"{num / 1000_000_000:.1f}B"
    elif num >= 1000000:
        return f"{num / 1000000:.1f}M"
    elif num >= 1000:
        return f"{num / 1000:.1f}K"
    else:
        return str(num)
def load_test_data(test_file, object_file):
    # 讀取 objects.json
    with open(object_file, 'r') as f:
        object_dict = json.load(f)

    # 讀取 test.json
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    # 初始化 one-hot encoding 的數組
    num_classes = len(object_dict)
    batch_size = len(test_data)
    one_hot_labels = np.zeros((batch_size, num_classes), dtype=np.float32)

    # 將 test.json 中的標籤轉換為 one-hot encoding
    for i, item_list in enumerate(test_data):
        for obj in item_list:
            if obj in object_dict:
                one_hot_labels[i][object_dict[obj]] = 1
        

    return torch.tensor(one_hot_labels)

class TrainDDPM:

    
    def __init__(self,args):
        self.args = args
        
        self.model = ClassConditionedUnet(num_classes=24,class_emb_size=self.args.class_emb_size).to(self.args.device)
        
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.args.timesteps, beta_schedule='squaredcos_cap_v2')

        self.criterion = nn.MSELoss()
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())

        self.evaluate = evaluation_model()

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

    def valid_one_epoch(self,test_file,object_file,show_noise=False,all_test=False):
        transform=transforms.Compose([
            transforms.Normalize((0, 0, 0), (2, 2, 2)),
            transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
        ])
        label = load_test_data(test_file, object_file)

        image_ls = []
        # sampling
        with torch.inference_mode():
            batch_size = label.shape[0]
            x = torch.rand(batch_size,3,64,64).to(self.args.device)
            if (batch_size != 1):
                label = label.squeeze()
            label = label.to(self.args.device,dtype=torch.float32)
            
            for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                with torch.no_grad():
                    residual = self.model(x,t,label)

                x = self.noise_scheduler.step(residual,t,x).prev_sample
                if (not all_test and i % 100 == 0):
                    img = transform(x)
                    self.save_images(img, name="ndenoising_"+str(i))
            
            ret = self.evaluate.eval(x, label)
            if (not all_test):
                print("ACC:",ret)
                img = transform(x)
                self.save_images(img, name="nfinal_test")
                
        return ret



    def save_model(self,epoch,suffix=None):
        os.makedirs("ddpm_ckpt",exist_ok=True)
        if (suffix != None):
            torch.save(self.model.state_dict(), os.path.join("ddpm_ckpt", f"ddpm_{epoch}_{suffix}.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join("ddpm_ckpt", f"ddpm_{epoch}.pth"))
    
    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))
    
    def all_test(self,path,class_emb_size):
        # path -> root
        model_files = sorted([f for f in os.listdir(path) if f.endswith('.pth')])
        print(model_files)
        model_files = [f for f in model_files if 'ddpm_' in f and (100 <= int(f.split('_')[1].split('.pth')[0]))]
        for model_file in model_files:
            epoch = int(model_file.split('_')[1].split('.pth')[0])
            model_path = os.path.join(path, model_file)
            # print(model_path)
            self.load_model(model_path)

            # do inference get accuracy
            accuracy = self.valid_one_epoch(self.args.test_source,"objects.json",self.args.all_test)
            print(epoch,accuracy)

            record_testing_result(epoch, accuracy, path,class_emb_size,file_name=f"{self.args.test_source}_accuracy_results.csv")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDPM")
    
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--num_workers', type=int, default=8)

    
    parser.add_argument('--ckpt_path', type=str,default="")

    parser.add_argument('--test_source', type=str,default="test.json")

    parser.add_argument('--timesteps', type=int,default=1000)
    parser.add_argument('--class_emb_size', type=int, default=512, help='embedding class size')
    parser.add_argument('--gpu_ids', type=str, default="0", help='gpuid')
    parser.add_argument('--ckpt_folder', type=str, default="", help='all testing folder')

    
    parser.add_argument('--all_test', action='store_true')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    

    ddpmTrainer = TrainDDPM(args)
    if (args.all_test and args.ckpt_folder):
        ddpmTrainer.all_test(args.ckpt_folder,args.class_emb_size)
    elif (args.ckpt_path != ""):
        ddpmTrainer.load_model(args.ckpt_path)
        ddpmTrainer.valid_one_epoch(args.test_source,"objects.json")
