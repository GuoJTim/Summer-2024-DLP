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
from datasets.BracketFlare_dataset import BracketFlare_Loader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from utils.UTILS import compute_psnr,MixUp_AUG,rand_bbox,compute_ssim
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from loss.perceptual import LossNetwork
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loss.contrastive_loss import HCRLoss
from networks.network_RefDet import RefDet,RefDetDual
import matplotlib.image as img

sys.path.append(os.getcwd())

if torch.cuda.device_count() ==8:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,2,3,4, 5,6,7"
    device_ids = [0, 1,2,3,4, 5,6,7]
if torch.cuda.device_count() == 4:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,2,3"
    device_ids = [0, 1,2,3]
if torch.cuda.device_count() == 2:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device_ids = [0, 1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:',device)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:', device)

parser = argparse.ArgumentParser()
# path setting
parser.add_argument('--experiment_name', type=str,
                    default="TEST")  # modify the experiments name-->modify all save path
parser.add_argument('--unified_path', type=str, default='/app/runs/')
# parser.add_argument('--model_save_dir', type=str, default= )#required=True
parser.add_argument('--training_data_path', type=str,
                    default='/gdata1/zhuyr/Deref/training_data/')
#parser.add_argument('--training_data_path_Txt', type=str, default='/gdata1/zhuyr/Deref/training_data/Ref_HZ1.txt')
parser.add_argument('--training_data_path_Txt', nargs='*', help='a list of strings')
parser.add_argument('--training_data_path_Txt1', nargs='*', help='a list of strings')
# --experiment_name SIRRwPreD --EPOCH 150 --T_period 50 --Crop_patches 320 --training_data_path_Txt '/mnt/data_oss/ReflectionData/SIRR_USTC/DeRef_USTC_wPreD.txt'

parser.add_argument('--writer_dir', type=str, default='/app/runs/logs/')

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


# parser.add_argument('--eval_in_path_SIR', type=str, default='/gdata1/zhuyr/Deref/training_data/SIR/blended/')
# parser.add_argument('--eval_gt_path_SIR', type=str, default='/gdata1/zhuyr/Deref/training_data/SIR/transmission_layer/')
# training setting
parser.add_argument('--EPOCH', type=int, default=200)
parser.add_argument('--T_period', type=int, default=50)  # CosineAnnealingWarmRestarts
parser.add_argument('--BATCH_SIZE', type=int, default=1)
parser.add_argument('--START_EPOCH', type=int, default=0) # args.START_EPOCH
parser.add_argument('--Crop_patches', type=int, default=320)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--learning_rate_Det', type=float, default=0.0002)


parser.add_argument('--print_frequency', type=int, default=1)
parser.add_argument('--SAVE_Inter_Results', type=bool, default=True)
parser.add_argument('--SAVE_test_Results', type=bool, default=True)
# during training
parser.add_argument('--max_psnr', type=int, default=10)
parser.add_argument('--fix_sample', type=int, default=100000)


parser.add_argument('--gamma_rr', type=float, default=0.1) ##
parser.add_argument('--gamma_rd', type=float, default=0.0001, help='max gamma in synthetic dataset') ##
parser.add_argument('--gamma_fd', type=float, default=0.0001, help='max gamma in synthetic dataset') ##


parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--RDNet_Receptual_Loss', type=str, default='VGG') # VGG RDNet_Receptual_Loss
parser.add_argument('--hue_loss', type=str2bool, default=False)
parser.add_argument('--lam_hueLoss', type=float, default=0.1)
parser.add_argument('--RRNet_Other_Loss', type=str, default='none')
parser.add_argument('--gamma_other', type=float, default=0.1)

parser.add_argument('--Aug_regular', type=str2bool, default=False)
parser.add_argument('--MixUp_AUG', type=str2bool, default=False)

# training setting
parser.add_argument('--base_channel', type=int, default=32)
parser.add_argument('--base_channel_refineNet', type=int, default=24)

parser.add_argument('--num_block', type=int, default=6)

parser.add_argument('--enc_blks', nargs='+', type=int, help='List of integers')
parser.add_argument('--dec_blks', nargs='+', type=int, help='List of integers')
parser.add_argument('--middle_blk_num', type=int, default=1)

parser.add_argument('--fusion_ratio', type=float, default=0.7)

parser.add_argument('--load_pre_model', type=str2bool, default= False) # VGG
parser.add_argument('--pre_model', type=str, default='None') # VGG
parser.add_argument('--pre_model1', type=str, default='None') # VGG
parser.add_argument('--pre_model2', type=str, default='None') # VGG

parser.add_argument('--pre_model_strict', type=str2bool, default= False) # VGG


parser.add_argument('--eval_freq', type=int, default=5) # epoch
 
# network structure

parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--hyper', type=str2bool, default=False)
parser.add_argument('--drop_flag', type=str2bool, default=False)
parser.add_argument('--drop_rate', type=float, default= 0.4)




parser.add_argument('--augM', type=str2bool, default=False)
parser.add_argument('--in_norm', type=str2bool, default=False)
parser.add_argument('--pyramid', type=str2bool, default=False)
parser.add_argument('--global_skip', type=str2bool, default=False)

parser.add_argument('--adjust_loader', type=str2bool, default=False)

# syn data
parser.add_argument('--low_sigma', type=float, default=2, help='min sigma in synthetic dataset')
parser.add_argument('--high_sigma', type=float, default=5, help='max sigma in synthetic dataset')
parser.add_argument('--low_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
parser.add_argument('--high_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')

parser.add_argument('--syn_mode', type=int, default=3)

parser.add_argument('--low_A', type=float, default=2, help='min sigma in synthetic dataset')
parser.add_argument('--high_A', type=float, default=5, help='max sigma in synthetic dataset')
parser.add_argument('--low_beta', type=float, default=1.3, help='max gamma in synthetic dataset')
parser.add_argument('--high_beta', type=float, default=1.3, help='max gamma in synthetic dataset')

# DDP
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument('--RRNet_Loss_func', type=str2bool, default=False)

# cutmix
parser.add_argument('--cutmix', type=str2bool, default=False, help='max gamma in synthetic dataset')
parser.add_argument('--cutmix_prob', type=float, default=0.5, help='max gamma in synthetic dataset')

parser.add_argument('--Det_model', type=str, default='None') # VGG

parser.add_argument('--concat', type=str2bool, default=True, help='merge manner')
parser.add_argument('--merge_manner', type=int, default= 0 )

parser.add_argument('--save_pth_model', type=str2bool, default=True)

parser.add_argument('--RDNet_Loss_func', type=str, default='None') ##

parser.add_argument('--FDNet_Loss_func', type=str, default='None') ##

parser.add_argument('--load_model_flag', type=int, default= 0 )


#  --in_norm   --pyramid
args = parser.parse_args()
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
print("local_rank",local_rank)


if args.debug == True:
    fix_sampleA = 50
    fix_sampleB = 50
    fix_sampleC = 50
    print_frequency = 5

else:
    fix_sampleA = args.fix_sample
    fix_sampleB = args.fix_sample
    fix_sampleC = args.fix_sample
    print_frequency = args.print_frequency

exper_name = args.experiment_name
writer = SummaryWriter(args.writer_dir + exper_name)
# if not os.path.exists(args.writer_dir):
#     os.mkdir(args.writer_dir)
os.makedirs(args.writer_dir,exist_ok=True)

unified_path = args.unified_path
SAVE_PATH = unified_path + exper_name + '/'
if not os.path.exists(SAVE_PATH):
    #os.mkdir(SAVE_PATH,)
    os.makedirs(SAVE_PATH,exist_ok=True)

if args.SAVE_Inter_Results:
    SAVE_Inter_Results_PATH = unified_path + exper_name + '__inter_results/'
    if not os.path.exists(SAVE_Inter_Results_PATH):
        #os.mkdir(SAVE_Inter_Results_PATH)
        os.makedirs(SAVE_Inter_Results_PATH, exist_ok=True)

trans_eval = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print("==" * 50)


def check_dataset(in_path, gt_path, name='RD'):
    print("Check {} pairs({}) ???: {} ".format(name, len(os.listdir(in_path)), os.listdir(in_path) == os.listdir(gt_path)))

check_dataset(args.eval_in_path_nature20, args.eval_gt_path_nature20, 'val-nature20')
check_dataset(args.eval_in_path_wild55, args.eval_gt_path_wild55, 'val-wild55')
check_dataset(args.eval_in_path_real20, args.eval_gt_path_real20, 'val-real20')
check_dataset(args.eval_in_path_postcard199, args.eval_gt_path_postcard199, 'val-postcard199')
check_dataset(args.eval_in_path_soild200, args.eval_gt_path_soild200, 'val-soild200')

print("==" * 50)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



def save_imgs_for_visual(path, inputs, labels, outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path, nrow=3, padding=0)
def save_imgs_for_visual4(path, inputs, labels, outputs,sparse_out):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0], sparse_out.cpu()[0]], path, nrow=4, padding=0)
def save_imgs_for_visualR2(path, inputs, labels, outputs, inputs1, labels1, outputs1):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0] ,
                                  inputs1.cpu()[0], labels1.cpu()[0], outputs1.cpu()[0] ], path, nrow=3, padding=0)


# ===========================================================================
#                                 TESTING
# ===========================================================================
def test(RRNet, RDNet,FDNet, eval_loader, Dname='S', SAVE_test_Results = False):
    RRNet.eval()
    FDNet.eval()
    RDNet.eval()
    iter_num = 0
    with torch.no_grad():
        eval_results ={'eval_input_psnr': 0.0, 'eval_output_psnr': 0.0,
                       'eval_input_ssim': 0.0, 'eval_output_ssim': 0.0,
                       'infer_time': 0.0}
        st = time.time()
        for index, (data_in, label, _,_,name) in enumerate(eval_loader, 0):  # enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            iter_num += 1
            infer_st = time.time()
            reflection_mask = RDNet(inputs)
            flare_mask = FDNet(inputs)
            sparse_out = torch.max(reflection_mask,flare_mask)
            #net_inputs = torch.cat([inputs, sparse_out], dim=1)
            # outputs = RRNet(inputs, sparse_out)
            outputs = RRNet(inputs, reflection_mask,flare_mask)


            eval_results['infer_time'] += time.time()-infer_st
            eval_input_psnr = compute_psnr(inputs, labels)
            eval_output_psnr = compute_psnr(outputs, labels)
            eval_input_ssim = compute_ssim(inputs, labels)
            eval_output_ssim = compute_ssim(outputs, labels)
            eval_results['eval_input_psnr'] += compute_psnr(inputs, labels)
            eval_results['eval_output_psnr'] += compute_psnr(outputs, labels)
            eval_results['eval_input_ssim'] += compute_ssim(inputs, labels)
            eval_results['eval_output_ssim'] += compute_ssim(outputs, labels)


            if SAVE_test_Results:
                SAVE_Test_Results_PATH = unified_path + exper_name + '__test_results/'
                os.makedirs(SAVE_Test_Results_PATH, exist_ok=True)
                Final_SAVE_Test_Results_PATH_img = SAVE_Test_Results_PATH  + Dname + '-img' + '/'
                os.makedirs(Final_SAVE_Test_Results_PATH_img, exist_ok=True)
                # save_imgs_for_visual4(
                #     Final_SAVE_Test_Results_PATH + name[0] + '.jpg',
                #     inputs, labels, outputs, sparse_out.repeat(1, 3, 1, 1))

                out_eval_np = np.squeeze(torch.clamp(outputs, 0., 1.).cpu().detach().numpy()).transpose((1, 2, 0))
                if (name[0] == ""):
                    fname = str(iter_num)+"_INPUT_PSNR_"+str(eval_input_psnr)+"_OUTPUT_PSNR_"+str(eval_output_psnr)+"_INPUT_SSIM_"+str(eval_input_ssim)+"_INPUT_SSIM_"+str(eval_output_ssim)
                else:
                    fname = name[0].split('.')[0]+"_INPUT_PSNR_"+str(eval_input_psnr)+"_OUTPUT_PSNR_"+str(eval_output_psnr)+"_INPUT_SSIM_"+str(eval_input_ssim)+"_INPUT_SSIM_"+str(eval_output_ssim)
                # print(Final_SAVE_Test_Results_PATH_img + fname + '.png')
                #img.imsave(Final_SAVE_Test_Results_PATH_img + fname + '.png', np.uint8(out_eval_np * 255.))
                save_imgs_for_visual(Final_SAVE_Test_Results_PATH_img + fname + '.png',
                                        inputs, 
                                        labels, 
                                        outputs)
                Final_SAVE_Test_Results_PATH_location = SAVE_Test_Results_PATH + Dname + '-location_gray' + '/'
                os.makedirs(Final_SAVE_Test_Results_PATH_location, exist_ok=True)

                location_eval_np = np.squeeze(torch.clamp(sparse_out, 0., 1.).cpu().detach().numpy())#.transpose((1, 2, 0))
                img.imsave(Final_SAVE_Test_Results_PATH_location + fname + '.png', np.uint8(location_eval_np * 255.), cmap='gray')

        Final_output_PSNR = eval_results['eval_output_psnr'] / len(eval_loader)
        Final_input_PSNR = eval_results['eval_input_psnr'] / len(eval_loader)
        Final_output_SSIM = eval_results['eval_output_ssim'] / len(eval_loader)
        Final_input_SSIM = eval_results['eval_input_ssim'] / len(eval_loader)


        print("Dname:{}-------[Num_eval:{} In_PSNR:{}  Out_PSNR:{} , In_SSIM:{}  Out_SSIM:{}], [total cost time: {} || total infer time:{} avg infer time:{} ]".format(
                    Dname, len(eval_loader), round(Final_input_PSNR, 4),
                    round(Final_output_PSNR, 4), round(Final_input_SSIM, 4),
                    round(Final_output_SSIM, 4), time.time() - st, eval_results['infer_time'] , eval_results['infer_time'] / len(eval_loader)))






def save_imgs_for_visual(path, inputs, labels, outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path, nrow=3, padding=0)
def save_imgs_for_visual4(path, inputs, labels, outputs,sparse_out):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0], sparse_out.cpu()[0]], path, nrow=4, padding=0)
def save_imgs_for_visualR2(path, inputs, labels, outputs, inputs1, labels1, outputs1,outputs2,outputs3,outputs4):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0] ,
                                  inputs1.cpu()[0], labels1.cpu()[0], outputs1.cpu()[0], 
                                  outputs2.cpu()[0],outputs3.cpu()[0],outputs4.cpu()[0] ], path, nrow=3, padding=0)




from torch.utils.data import Dataset, ConcatDataset

from datasets.image_folder import read_fns


# ===========================================================================
#                                 TRAINING DATA
# ===========================================================================
def get_training_data(fix_sampleA=fix_sampleA, Crop_patches=args.Crop_patches):
    rrw_opt = {
        "rrw_root": '/app/RRW/',
        "rrw_txt": './gt_in_pairs_v2.txt',
        "image_size": 256,   # image size
        "sample_size": 500,  # how many image will use
        "augmentation": True, # augmentation
    }

    reflection_dataset = my_dataset_wTxt(
        rrw_root=rrw_opt["rrw_root"],
        rrw_txt=rrw_opt["rrw_txt"],
        image_size=rrw_opt["image_size"],
        sample_size=rrw_opt["sample_size"],
        augmentation=rrw_opt["augmentation"],
    )

    flare_opt = {
        'background_path': '/app/BracketFlare/gt',
        'flare_path': '/app/BracketFlare/flare',
        'mask_type': 'flare',
        'img_size': 256,
        'translate': 10/4000,
        'preprocess_size': 256,
        'background_color': 0.03
    }
    flare_dataset = BracketFlare_Loader(flare_opt)
    trans_eval = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    train_nature200 = my_dataset_eval(
        root_in="/app/train_nature/blended",
        root_label="/app/train_nature/transmission_layer",
        transform=trans_eval,
    )

    train_real89 = my_dataset_eval(
        root_in="/app/train_real/blended",
        root_label="/app/train_real/transmission_layer",
        transform=trans_eval,
    )

    hybrid_datasets = ConcatDataset([reflection_dataset, flare_dataset, train_nature200, train_real89])

    train_sampler = torch.utils.data.distributed.DistributedSampler(hybrid_datasets)

    train_loader = DataLoader(
        dataset=hybrid_datasets,
        batch_size=args.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        sampler=train_sampler,
    )
    
    print('len(train_loader):', len(train_loader))
    return train_loader

# ===========================================================================
#                                 GET OTHER DATASET
# ===========================================================================
def get_eval_data(val_in_path=args.eval_in_path_nature20, val_gt_path=args.eval_gt_path_nature20
                  , trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label=val_gt_path, transform=trans_eval, fix_sample=500)

    # eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_data)
    # eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4,sampler=eval_sampler)

    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=0)
    return eval_loader


def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))


from collections import OrderedDict

if __name__ == '__main__':

    print('====='*10,'from networks.NAFNet_arch import NAFNet, NAFNetLocal')
    from networks.NAFNet_arch import NAFNet_wDetHead
    if args.hyper:
        img_channel = args.img_channel + 1472
    else:
        img_channel = args.img_channel

    NAFNet = NAFNet_wDetHead(img_channel= 3, width=args.base_channel, middle_blk_num=args.middle_blk_num,
                      enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks, global_residual=args.global_skip,
                     drop_flag = args.drop_flag,  drop_rate=args.drop_rate,
                          concat = args.concat, merge_manner = args.merge_manner)
    RDNet = RefDet(backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 num_of_layers=6,
                 expansion = 4)
    
    FDNet = RefDet(backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 num_of_layers=6,
                 expansion = 4)

    #Det_model  =
    #net_Det.load_state_dict(torch.load(args.Det_model), strict=True)

    # 第一种加载之前dp训练的可以 这个ddp不行
    # if args.load_pre_model:
    #     checkpoint = torch.load(args.pre_model, map_location='cuda:{}'.format(local_rank))
    #     net.load_state_dict(checkpoint, strict= True)



    # 第二种直接加载呢
    if args.load_pre_model:
        checkpoint = torch.load(args.pre_model, map_location='cuda:{}'.format(local_rank))
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                name = k[7:]  # 去掉 'module.' 前缀
            else:
                name = k  # 保持原样
            new_state_dict[name] = v
        NAFNet.load_state_dict(new_state_dict, strict= args.pre_model_strict)
    
        checkpoint1 = torch.load(args.pre_model1, map_location='cuda:{}'.format(local_rank))
        new_state_dict1 = OrderedDict()
        for k, v in checkpoint1.items():
            if k.startswith('module.'):
                name = k[7:]  # 去掉 'module.' 前缀
            else:
                name = k  # 保持原样
            new_state_dict1[name] = v
        RDNet.load_state_dict(new_state_dict1, strict=True)

        checkpoint2 = torch.load(args.pre_model2, map_location='cuda:{}'.format(local_rank))
        new_state_dict2 = OrderedDict()
        for k, v in checkpoint2.items():
            if k.startswith('module.'):
                name = k[7:]  # 去掉 'module.' 前缀
            else:
                name = k  # 保持原样
            new_state_dict2[name] = v
        FDNet.load_state_dict(new_state_dict2, strict=True)

    # if args.load_pre_model and (args.load_model_flag == 0):
    #     checkpoint = torch.load(args.pre_model)
    #     NAFNet.load_state_dict(checkpoint, strict=True)

    #     print('--'*200,'sucess!  load pre-model (NAFNet)  ')
    #     checkpoint1 = torch.load(args.pre_model1)
    #     RDNet.load_state_dict(checkpoint1, strict=True)
    #     print('=='*200,'sucess!  load pre-model (RDNet)  ')

    #     checkpoint2 = torch.load(args.pre_model2)
    #     FDNet.load_state_dict(checkpoint2, strict=True)
    #     print('=='*200,'sucess!  load pre-model (FDNet)  ')

    FDNet.to(local_rank)
    FDNet = DDP(FDNet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    RDNet.to(local_rank)
    RDNet = DDP(RDNet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    NAFNet.to(local_rank)
    NAFNet = DDP(NAFNet, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    print_param_number(NAFNet)
    print_param_number(FDNet)
    print_param_number(RDNet)

    # optimizerG_Det = optim.Adam(net_Det.parameters(),
    #                         lr=args.learning_rate_Det, betas=(0.9, 0.999))
    # scheduler_Det = CosineAnnealingWarmRestarts(optimizerG_Det, T_0=args.T_period, T_mult=1)


# ===========================================================================
#                                 DIFFERENT DATASET EVAL
# ===========================================================================
            


    eval_loader_nature20 = get_eval_data(val_in_path=args.eval_in_path_nature20, val_gt_path=args.eval_gt_path_nature20)
    eval_loader_real20 = get_eval_data(val_in_path=args.eval_in_path_real20, val_gt_path=args.eval_gt_path_real20)
    eval_loader_wild55 = get_eval_data(val_in_path=args.eval_in_path_wild55, val_gt_path=args.eval_gt_path_wild55)
    eval_loader_postcard199 = get_eval_data(val_in_path=args.eval_in_path_postcard199, val_gt_path=args.eval_gt_path_postcard199)
    eval_loader_soild200 = get_eval_data(val_in_path=args.eval_in_path_soild200, val_gt_path=args.eval_gt_path_soild200)
    # eval_loader_SIR = get_eval_data(val_in_path=args.eval_in_path_SIR, val_gt_path=args.eval_gt_path_SIR)


    rrw_opt = {
        "rrw_root": '/app/RRW/',
        "rrw_txt": './gt_in_pairs_v2.txt',
        "image_size": 256,   # image size
        "sample_size": 500,  # how many image will use
        "augmentation": True, # augmentation
    }

    reflection_dataset = my_dataset_wTxt(
        rrw_root=rrw_opt["rrw_root"],
        rrw_txt=rrw_opt["rrw_txt"],
        image_size=rrw_opt["image_size"],
        sample_size=rrw_opt["sample_size"],
        augmentation=rrw_opt["augmentation"],
    )
    eval_loader_reflection = DataLoader(dataset=reflection_dataset, batch_size=1, num_workers=0)

    flare_dataset = my_dataset_eval(
        root_in="/app/BracketFlare/test/lq",
        root_label="/app/BracketFlare/test/gt",
        transform=trans_eval,
    )
    

    eval_loader_flare = DataLoader(dataset=flare_dataset, batch_size=1, num_workers=0)



    trans_eval = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    nature200_dataset = my_dataset_eval(
        root_in="/app/train_nature/blended",
        root_label="/app/train_nature/transmission_layer",
        transform=trans_eval,
    )

    real89_dataset = my_dataset_eval(
        root_in="/app/train_real/blended",
        root_label="/app/train_real/transmission_layer",
        transform=trans_eval,
    )

    eval_loader_nature200 = DataLoader(dataset=nature200_dataset, batch_size=1, num_workers=0)
    eval_loader_real89 = DataLoader(dataset=real89_dataset, batch_size=1, num_workers=0)


    # train data 
    test(RRNet=NAFNet, RDNet =RDNet,FDNet=FDNet,  
        eval_loader=eval_loader_flare, 
        Dname='flare',SAVE_test_Results =True)


    # test(RRNet=NAFNet, RDNet =RDNet,FDNet=FDNet,  
    #     eval_loader=eval_loader_nature200, 
    #     Dname='train_nature200',SAVE_test_Results =True)


    # test(RRNet=NAFNet, RDNet =RDNet,FDNet=FDNet,  
    #     eval_loader=eval_loader_real89, 
    #     Dname='train_real89',SAVE_test_Results =True)





    # https://github.com/JHL-HUST/IBCLN?tab=readme-ov-file
    test(RRNet=NAFNet, RDNet =RDNet,FDNet=FDNet,
        eval_loader=eval_loader_nature20, 
        Dname='nature20',SAVE_test_Results =True)

    # https://github.com/ceciliavision/perceptual-reflection-removal
    test(RRNet=NAFNet, RDNet =RDNet,FDNet=FDNet,  
        eval_loader=eval_loader_real20, 
        Dname='real20',SAVE_test_Results =True)

    

                                # SIRR  
    test(RRNet=NAFNet, RDNet =RDNet,FDNet=FDNet,  
        eval_loader=eval_loader_wild55, 
        Dname='wild55',SAVE_test_Results =True)

    test(RRNet=NAFNet, RDNet =RDNet,FDNet=FDNet, 
        eval_loader=eval_loader_postcard199 , 
        Dname='postcard199',SAVE_test_Results =True)

    test(RRNet=NAFNet, RDNet =RDNet,FDNet=FDNet,  
        eval_loader=eval_loader_soild200, 
        Dname='soild200',SAVE_test_Results =True)




