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

from utils.UTILS import compute_psnr,MixUp_AUG,rand_bbox
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from loss.perceptual import LossNetwork
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loss.contrastive_loss import HCRLoss
from networks.network_RefDet import RefDet,RefDetDual


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
                    default="SSIR_ONLY")  # modify the experiments name-->modify all save path
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



# ===========================================================================
#                                 TESTING
# ===========================================================================
def test(RRNet, RDNet, eval_loader, epoch=1,iters=100, max_psnr_val=19, Dname='S', SAVE_test_Results = False):
    RRNet.eval()
    RDNet.eval()

    with torch.no_grad():
        eval_output_psnr = 0.0
        eval_input_psnr = 0.0
        st = time.time()
        for index, (data_in, label, name,_) in enumerate(eval_loader, 0):  # enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)

            reflection_mask = RDNet(inputs)
            #net_inputs = torch.cat([inputs, sparse_out], dim=1)
            # outputs = RRNet(inputs, sparse_out)
            outputs = RRNet(inputs,reflection_mask)


            eval_input_psnr += compute_psnr(inputs, labels)
            eval_output_psnr += compute_psnr(outputs, labels)

            if index < 10 and SAVE_test_Results:
                SAVE_Test_Results_PATH = unified_path + exper_name + '__test_results/'
                os.makedirs(SAVE_Test_Results_PATH, exist_ok=True)

                # if not os.path.exists(SAVE_Test_Results_PATH):
                #     os.mkdir(SAVE_Test_Results_PATH)
                Final_SAVE_Test_Results_PATH =   SAVE_Test_Results_PATH  + Dname + '/'
                # if not os.path.exists(Final_SAVE_Test_Results_PATH):
                #     os.mkdir(Final_SAVE_Test_Results_PATH)
                os.makedirs(Final_SAVE_Test_Results_PATH, exist_ok=True)

                save_imgs_for_visual4(Final_SAVE_Test_Results_PATH + name[0] + '-'+str(epoch) +'_'+ str(iters)  + '.jpg',
                                     inputs, labels, outputs, sparse_out.repeat(1, 3, 1, 1) )

        Final_output_PSNR = eval_output_psnr / len(eval_loader)
        Final_input_PSNR = eval_input_psnr / len(eval_loader)
        
        writer.add_scalars(exper_name + '/testing', {
            'eval_PSNR_Output': eval_output_psnr / len(eval_loader),
            'eval_PSNR_Input': eval_input_psnr / len(eval_loader), 
        }, epoch)

        if Final_output_PSNR > max_psnr_val:
            max_psnr_val = Final_output_PSNR
        print(
            "epoch:{}------Dname:{}-------[Num_eval:{} In_PSNR:{}  Out_PSNR:{}]--------max_psnr_val:{}, cost time: {}".format(
                epoch, Dname, len(eval_loader), round(Final_input_PSNR, 2),
                round(Final_output_PSNR, 2), round(max_psnr_val, 2), time.time() - st))

    return max_psnr_val


def save_imgs_for_visual(path, inputs, labels, outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path, nrow=3, padding=0)
def save_imgs_for_visual4(path, inputs, labels, outputs,sparse_out):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0], sparse_out.cpu()[0]], path, nrow=4, padding=0)
def save_imgs_for_visualR2(path, inputs, labels, outputs, inputs1, labels1, outputs1):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0] ,
                                  inputs1.cpu()[0], labels1.cpu()[0], outputs1.cpu()[0] ], path, nrow=3, padding=0)




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
    
    hybrid_datasets = ConcatDataset([reflection_dataset, flare_dataset])
    
    
    hybrid_datasets = reflection_dataset
    
    
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(hybrid_datasets)

    train_loader = DataLoader(dataset=hybrid_datasets, batch_size=args.BATCH_SIZE,
                              num_workers= 0 ,shuffle=False, sampler=train_sampler
                              )
    
    print('len(train_loader):', len(train_loader))
    return train_loader
# def get_training_data_re(fix_sampleA=fix_sampleA, Crop_patches=args.Crop_patches):
#     rootA = args.training_data_path

#     rootA_txt1_list = args.training_data_path_Txt1
#     train_Pre_dataset_list = []
#     for idx_dataset in range(len(rootA_txt1_list)):
#         train_Pre_dataset = my_dataset_wTxt(rootA, rootA_txt1_list[idx_dataset],
#                                             crop_size=Crop_patches,
#                                             fix_sample_A=fix_sampleA,
#                                             regular_aug=args.Aug_regular)  # threshold_size =  args.threshold_size
#         train_Pre_dataset_list.append(train_Pre_dataset)
#     train_pre_datasets = ConcatDataset(train_Pre_dataset_list)

#     datadir_syn = '/gdata1/zhuyr/Deref/training_data/JPEGImages'
#     train_dataset_syn = CEILDataset(
#         datadir_syn, read_fns('/ghome/zhuyr/ADeref_two1/VOC2012_224_train_png.txt'), size=None,
#         enable_transforms=True,
#         low_sigma=args.low_sigma, high_sigma=args.high_sigma,
#         low_gamma=args.low_gamma, high_gamma=args.high_gamma,crop_size=args.Crop_patches, mode=args.syn_mode,
#                  low_A=args.low_A, high_A=args.high_A,
#                 low_beta=args.low_beta, high_beta=args.high_beta)

#     train_set = FusionDataset([train_dataset_syn, train_pre_datasets], fusion_ratios=[0.5, 0.5] )#fusion_ratios=[args.fusion_ratio,1.0 - args.fusion_ratio])

#     train_loader = DataLoader(dataset=train_set, batch_size=args.BATCH_SIZE, num_workers=8, shuffle=True)
#     print('len(train_loader):', len(train_loader))
#     return train_loader

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

# ===========================================================================
#                                 EROSION DILATION
# ===========================================================================
def get_erosion_dilation(input_mask, flag=True, kernel_size=3):
    # B*3*H*W, B*1*H*W
    if flag: #flag True 腐蚀操作
        temp_mask = -F.max_pool2d(-input_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return temp_mask
    else: # Flase 操作
        return F.max_pool2d(input_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

# ===========================================================================
#                                 MAXRF formula
# ===========================================================================
# def obtain_sparse_reprentation(tensorA, tensorB):
#     maxA = tensorA.max(dim=1)[0]
#     maxB = tensorB.max(dim=1)[0]
#     # 定义 sobel 滤波器
#     #sobel_filter = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

#     sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#     sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)


#     A_grad_x = F.conv2d(maxA.unsqueeze(1), sobel_x, padding=1)
#     A_grad_y = F.conv2d(maxA.unsqueeze(1), sobel_y, padding=1)
#     grad1 = torch.sqrt(A_grad_x ** 2 + A_grad_y ** 2)

#     B_grad_x = F.conv2d(maxB.unsqueeze(1), sobel_x, padding=1)
#     B_grad_y = F.conv2d(maxB.unsqueeze(1), sobel_y, padding=1)
#     grad2 = torch.sqrt(B_grad_x ** 2 + B_grad_y ** 2)

#     # 比较 grad1 和 grad2 的值，如果 grad1 大于 grad2 的位置记为 1，其他设置为 0
#     mask = (grad1 > grad2).float()
#     return  mask

def obtain_sparse_reprentation(tensorA, tensorB):
    maxA = tensorA.max(dim=1)[0]
    maxB = tensorB.max(dim=1)[0]
    # 定义 sobel 滤波器
    sobel_filter = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    # 使用 sobel 滤波器求梯度
    grad1 = F.conv2d(maxA.unsqueeze(1), sobel_filter, padding=1)
    grad2 = F.conv2d(maxB.unsqueeze(1), sobel_filter, padding=1)

    # 比较 grad1 和 grad2 的值，如果 grad1 大于 grad2 的位置记为 1，其他设置为 0
    mask = (grad1 > grad2).float()
    return  mask

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
    # if args.load_pre_model:
    #     checkpoint = torch.load(args.pre_model, map_location='cuda:{}'.format(local_rank))
    #     new_state_dict = OrderedDict()
    #     for k, v in checkpoint.items():
    #         name = k[7:]  # module字段在最前面，从第7个字符开始就可以去掉module
    #         new_state_dict[name] = v  # 新字典的key值对应的value一一对应
    #     net.load_state_dict(new_state_dict, strict= args.pre_model_strict)
    #
    #     checkpoint1 = torch.load(args.pre_model1, map_location='cuda:{}'.format(local_rank))
    #     new_state_dict1 = OrderedDict()
    #     for k, v in checkpoint1.items():
    #         name = k[7:]  # module字段在最前面，从第7个字符开始就可以去掉module
    #         new_state_dict1[name] = v  # 新字典的key值对应的value一一对应
    #     net_Det.load_state_dict(new_state_dict1, strict=True)

    if args.load_pre_model and (args.load_model_flag == 0):
        checkpoint = torch.load(args.pre_model)
        NAFNet.load_state_dict(checkpoint, strict=True)

        print('--'*200,'sucess!  load pre-model (NAFNet)  ')
        checkpoint1 = torch.load(args.pre_model1)
        RDNet.load_state_dict(checkpoint1, strict=True)
        print('=='*200,'sucess!  load pre-model (RDNet)  ')

        checkpoint2 = torch.load(args.pre_model2)
        FDNet.load_state_dict(checkpoint2, strict=True)
        print('=='*200,'sucess!  load pre-model (FDNet)  ')

    RDNet.to(local_rank)
    RDNet = DDP(RDNet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    NAFNet.to(local_rank)
    NAFNet = DDP(NAFNet, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

    

    VGG19 = losses.Vgg19(requires_grad=False).to(local_rank)


    print_param_number(NAFNet)

    train_loader1 = get_training_data()
    #train_loader1_re = get_training_data()

    eval_loader_nature20 = get_eval_data(val_in_path=args.eval_in_path_nature20, val_gt_path=args.eval_gt_path_nature20)
    eval_loader_real20 = get_eval_data(val_in_path=args.eval_in_path_real20, val_gt_path=args.eval_gt_path_real20)
    eval_loader_wild55 = get_eval_data(val_in_path=args.eval_in_path_wild55, val_gt_path=args.eval_gt_path_wild55)
    eval_loader_postcard199 = get_eval_data(val_in_path=args.eval_in_path_postcard199, val_gt_path=args.eval_gt_path_postcard199)
    eval_loader_soild200 = get_eval_data(val_in_path=args.eval_in_path_soild200, val_gt_path=args.eval_gt_path_soild200)
    # eval_loader_SIR = get_eval_data(val_in_path=args.eval_in_path_SIR, val_gt_path=args.eval_gt_path_SIR)

    # net.parameters()
    optimizerG = optim.Adam([{'params': NAFNet.parameters(), 'lr': args.learning_rate},
                             {'params': RDNet.parameters(),'lr': args.learning_rate/2},
                             {'params': FDNet.parameters(),'lr': args.learning_rate/2}] ,
                            betas=(0.9, 0.999))# lr=args.learning_rate,
    scheduler = CosineAnnealingWarmRestarts(optimizerG, T_0=args.T_period, T_mult=1)  # ExponentialLR(optimizerG, gamma=0.98)

    # optimizerG_Det = optim.Adam(net_Det.parameters(),
    #                         lr=args.learning_rate_Det, betas=(0.9, 0.999))
    # scheduler_Det = CosineAnnealingWarmRestarts(optimizerG_Det, T_0=args.T_period, T_mult=1)


# ===========================================================================
#                                 LOSS FUNCTIONS
# ===========================================================================


    # Losses
    loss_char = losses.CharbonnierLoss()
    if args.hue_loss:
        criterion_hue = losses.HSVLoss()

    if args.RRNet_Other_Loss.lower() == 'hue':
        criterion = losses.HSVLoss()
    elif args.RRNet_Other_Loss.lower() == 'ssim':
        criterion = losses.SSIMLoss()
    elif args.RRNet_Other_Loss.lower() == 'contrast':
        criterion = HCRLoss()

    # 1
    # vgg = models.vgg16(pretrained=False)
    # 2
    if args.RDNet_Receptual_Loss == 'VGG':
        vgg = models.vgg16(pretrained=False)
        vgg.load_state_dict(torch.load('./networks/vgg/vgg16-397923af.pth'))
        vgg_model = vgg.features[:16]
        vgg_model = vgg_model.to(local_rank)
        for param in vgg_model.parameters():
            param.requires_grad = False
        loss_network = LossNetwork(vgg_model)
        loss_network.eval()


    step = 0
    max_psnr_val_nature20 = args.max_psnr
    max_psnr_val_real20 = args.max_psnr
    max_psnr_val_wild55 = args.max_psnr
    max_psnr_val_postcard199 = args.max_psnr
    max_psnr_val_soild200 = args.max_psnr
    max_psnr_val_SIR = args.max_psnr

    training_results ={ 
        'Total_loss':0.0 , 
        'RRNet_loss':0.0 , 
        'RRNet_L1_Loss':0.0 ,
        'RRNet_Loss2':0.0 , 
        'RRNet_Loss3':0.0 , 
        'RDNet_loss':0.0 ,
        'RDNet_L1_Loss':0.0 , 
        'RDNet_L1_Loss2':0.0 , 
        'FDNet_loss':0.0 , 
        'FDNet_L1_Loss':0.0 , 
        'FDNet_L1_Loss2':0.0 , 
        'input_PSNR_all':0.0 , 
        'train_PSNR_all':0.0 ,
    }

    Frequncy_eval_save = len(train_loader1)
# ===========================================================================
#                                 TRAINING
# ===========================================================================
    iter_nums = 0
    for epoch in range(args.EPOCH):

        st = time.time()
        if args.adjust_loader:
            if epoch < int(args.EPOCH * 0.7):
                train_loader = train_loader1
            else:
                train_loader = train_loader1_re
        else:
            train_loader = train_loader1

        scheduler.step(epoch)
        #scheduler_Det.step(epoch)

        train_loader.sampler.set_epoch(epoch)



        for i, train_data in enumerate(train_loader, 0):
            inputs, label, img_name, isFlare = train_data
            # Assuming isFlare is a tensor with shape [batch_size, 1] and contains boolean or 0/1 values
            batch_size = isFlare.size(0)
            w = inputs.size(2)
            h = inputs.size(3)

            data_sparse = get_erosion_dilation(obtain_sparse_reprentation(inputs, label))
            #get_erosion_dilation

            
            


            data_in = inputs
            label = label
            #print(i,"---------------Check data: data.size: {} ,in_GT_mask: {}".format(data_in.size(), label.size()))
            #if i == 0:
            #    print("Check data: data.size: {} ,in_GT_mask: {}".format(data_in.size(), label.size()))
            iter_nums += 1
            NAFNet.train()
            NAFNet.zero_grad()
            optimizerG.zero_grad()

            RDNet.train()
            RDNet.zero_grad()

            #optimizerG_Det.zero_grad()

            
            inputs = Variable(data_in).to(local_rank)
            labels = Variable(label).to(local_rank)
            labels_sparse = Variable(data_sparse).to(local_rank)


            r = np.random.rand()
            if args.cutmix and (r < args.cutmix_prob):
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(inputs.size()[0]).to(local_rank) #cuda()
                # target_a = target
                # target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1: bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                labels[:, :, bbx1: bbx2, bby1:bby2] = labels[rand_index, :, bbx1:bbx2, bby1:bby2]

                sparse_out = RDNet(inputs)
                #net_inputs = torch.cat([inputs,  ], dim=1)
                # train_output = NAFNet(inputs, sparse_out.detach())
                train_output = NAFNet(inputs, sparse_out)
            else:
                sparse_out = RDNet(inputs)
                # print(reflection_mask.shape)
                # print(flare_mask.shape)
                

                #net_inputs = torch.cat([inputs, sparse_out], dim=1) # .detach()
                # train_output = NAFNet(inputs, sparse_out.detach())
                train_output = NAFNet(inputs, sparse_out.detach())


            input_PSNR = compute_psnr(inputs, labels)
            trian_PSNR = compute_psnr(train_output, labels)
# ===========================================================================
#                                 RD LOSS
# ===========================================================================
            if args.RDNet_Loss_func.lower() =='char':
                RDNet_L1_Loss = loss_char(sparse_out, labels_sparse)
            else:
                RDNet_L1_Loss = losses.sigmoid_mse_loss(sparse_out, labels_sparse)
            RDNet_L1_Loss2 = losses.TVLoss(args.gamma_rd)(sparse_out)
            RDNet_loss =  RDNet_L1_Loss + RDNet_L1_Loss2 # RD Loss

# ===========================================================================
#                                 FD LOSS
# ===========================================================================
            # if args.FDNet_Loss_func.lower() =='char':
            #     FDNet_L1_Loss = loss_char(flare_mask, flare_labels_sparse)
            # else:
            #     FDNet_L1_Loss = losses.sigmoid_mse_loss(flare_mask, flare_labels_sparse)
            # FDNet_L1_Loss2 = losses.TVLoss(args.gamma_fd)(reflection_mask)
            # FDNet_loss = FDNet_L1_Loss + FDNet_L1_Loss2

# ===========================================================================
#                                 RR LOSS
# ===========================================================================
            if args.RRNet_Loss_func:
                RRNet_L1_Loss = loss_char(train_output, labels)  #F.smooth_l1_loss(train_output, labels)
            else:
                RRNet_L1_Loss = F.smooth_l1_loss(train_output, labels)
            if args.RDNet_Receptual_Loss == 'VGG':
                RRNet_Loss2 = args.gamma_rr * loss_network(train_output, labels)
            elif args.RDNet_Receptual_Loss == 'FFT':
                RRNet_Loss2 = args.gamma_rr * losses.fftLoss()(train_output, labels)
            else:
                RRNet_Loss2 = 0.01 * RRNet_L1_Loss

            if args.RRNet_Other_Loss.lower() == 'none':
                RRNet_Loss3 = RRNet_L1_Loss
                RRNet_loss = RRNet_L1_Loss + RRNet_Loss2
            elif args.RRNet_Other_Loss.lower() == 'contrast':
                RRNet_Loss3 = args.gamma_other * HCRLoss(train_output, labels, inputs)
                RRNet_loss = RRNet_L1_Loss + RRNet_Loss2 + RRNet_Loss3
            else:
                RRNet_Loss3 = args.gamma_other * criterion(train_output, labels)
                RRNet_loss = RRNet_L1_Loss + RRNet_Loss2 + RRNet_Loss3

            # RRNet_loss = RR Loss
            # RDNet_loss = RD Loss
            # FDNet_loss = FD Loss




            Total_loss  = RRNet_loss + RDNet_loss # + FDNet_loss

            training_results['Total_loss']     += Total_loss.item()
            training_results['RRNet_loss']     += RRNet_loss.item()
            training_results['RRNet_L1_Loss']     += RRNet_L1_Loss.item()
            training_results['RRNet_Loss2']     += RRNet_Loss2.item()
            training_results['RRNet_Loss3']     += RRNet_Loss3.item()


            training_results['RDNet_loss']     += RDNet_loss.item()
            training_results['RDNet_L1_Loss']     += RDNet_loss.item()
            training_results['RDNet_L1_Loss2']     += RDNet_loss.item()


            # training_results['FDNet_loss']     += FDNet_loss.item()
            # training_results['FDNet_L1_Loss']     += FDNet_loss.item()
            # training_results['FDNet_L1_Loss2']     += FDNet_loss.item()



            training_results['input_PSNR_all'] += input_PSNR
            training_results['train_PSNR_all'] += trian_PSNR

            #RDNet_loss.backward(retain_graph=True)
            #optimizerG_Det.step()

            Total_loss.backward()
            optimizerG.step()



            if (i + 1) % print_frequency == 0 : #and i > 1
                writer.add_scalars(exper_name + '/training', 
                                    {'PSNR_Output': training_results['train_PSNR_all'] / iter_nums,
                                    'PSNR_Input': training_results['input_PSNR_all'] / iter_nums, 
                                    }, iter_nums)
                writer.add_scalars(exper_name + '/training',
                                   {'Total_loss': training_results['Total_loss'] / iter_nums, 
                                   'RRNet_loss': training_results['RRNet_loss'] / iter_nums,
                                    'RDNet_loss': training_results['RDNet_loss'] / iter_nums, 
                                    #'FDNet_loss': training_results['FDNet_loss'] / iter_nums, 
                                    }, iter_nums)
                
                writer.add_scalars(exper_name + '/training',
                                   {
                                    'RRNet_loss': training_results['RRNet_loss'] / iter_nums, 
                                    'RRNet_L1_Loss': training_results['RRNet_L1_Loss'] / iter_nums,
                                    'RRNet_Loss2': training_results['RRNet_Loss2'] / iter_nums, 
                                    'RRNet_Loss3': training_results['RRNet_Loss3'] / iter_nums, 
                                    }, iter_nums)
                
                writer.add_scalars(exper_name + '/training',
                                   {
                                    'RDNet_loss': training_results['RDNet_loss'] / iter_nums, 
                                    'RDNet_L1_Loss': training_results['RDNet_L1_Loss'] / iter_nums,
                                    'RDNet_L1_Loss2': training_results['RDNet_L1_Loss2'] / iter_nums, 
                                    }, iter_nums)

                # writer.add_scalars(exper_name + '/training',
                #                    {'FDNet_loss': training_results['FDNet_loss'] / iter_nums, 
                #                    'FDNet_L1_Loss': training_results['FDNet_L1_Loss'] / iter_nums,
                #                     'FDNet_L1_Loss2': training_results['FDNet_L1_Loss2'] / iter_nums, 
                #                     }, iter_nums)

# epoch:%d,[%d / %d], [lr: %.7f ],[loss:%.5f,loss1:%.5f,loss2:%.5f,loss3:%.5f, ,avg_loss:%.5f || S1_loss:%.5f,S1_TV_loss:%.5f, avg_S1_loss:%.5f ],[in_PSNR: %.3f, out_PSNR: %.3f],time:%.3f
                info = [
                    f"epoch:{epoch},[{i+1:4} / {len(train_loader)}]",  
                    f"[lr:{optimizerG.param_groups[0]['lr']:.7f} ]", 
                    f"[RRNet_loss:{RRNet_loss.item():.5f},RDNet_loss:{RDNet_loss.item():.5f},avg_loss:{training_results['Total_loss'] / iter_nums:.5f} ",
                    f",in_PSNR:{input_PSNR:7.3f},out_PSNR:{trian_PSNR:7.3f}],time:{ time.time() - st :.3f}"
                ]
                print("".join(info))
            if (i + 1) % (print_frequency * 5) == 0 and i > 1:
                if args.SAVE_Inter_Results:
                    save_path = SAVE_Inter_Results_PATH + str(iter_nums) + '.jpg'
                    print(SAVE_Inter_Results_PATH)
                    #            reflection_labels_sparse = torch.empty_like(data_sparse).to(local_rank)
                    #          flare_labels_sparse =  torch.zeros(size=(batch_size,1,w,h)).to(local_rank)
                    save_imgs_for_visualR2(save_path, inputs, labels, train_output, data_sparse.repeat(1, 3, 1, 1),
                                           sparse_out.repeat(1, 3, 1, 1), F.sigmoid(sparse_out).repeat(1, 3, 1, 1))


        if ((epoch+1) % args.eval_freq) == 0:
            if dist.get_rank() == 0:
                if args.save_pth_model:
                    save_RR_model = SAVE_PATH + 'NAFNet_epoch_{}pth'.format(epoch+1)
                    torch.save(NAFNet.module.state_dict(), save_RR_model)
                    save_RD_model = SAVE_PATH + 'RDNet_epoch_{}.pth'.format(epoch+1)
                    torch.save(RDNet.module.state_dict(), save_RD_model)
                    #save_FD_model = SAVE_PATH + 'FDNet_epoch_{}.pth'.format(epoch+1)
                    #torch.save(FDNet.module.state_dict(), save_FD_model)

# ===========================================================================
#                                 DIFFERENT DATASET EVAL
# ===========================================================================
            
            # https://github.com/JHL-HUST/IBCLN?tab=readme-ov-file
            max_psnr_val_nature20 = test(RRNet=NAFNet, RDNet =RDNet,
                                        eval_loader=eval_loader_nature20, epoch=epoch, iters=iter_nums,
                                        max_psnr_val=max_psnr_val_nature20, Dname='nature20',SAVE_test_Results =True)

            # https://github.com/ceciliavision/perceptual-reflection-removal
            max_psnr_val_real20 = test(RRNet=NAFNet, RDNet =RDNet,  
                                        eval_loader=eval_loader_real20, epoch=epoch, iters=iter_nums,
                                        max_psnr_val=max_psnr_val_real20, Dname='real20',SAVE_test_Results =True)

            

                                        # SIRR  
            max_psnr_val_wild55 = test(RRNet=NAFNet, RDNet =RDNet,  
                                        eval_loader=eval_loader_wild55, epoch=epoch, iters=iter_nums,
                                        max_psnr_val=max_psnr_val_wild55, Dname='wild55',SAVE_test_Results =True)
            max_psnr_val_postcard199  = test(RRNet=NAFNet, RDNet =RDNet, 
                                        eval_loader=eval_loader_postcard199 , epoch=epoch,iters=iter_nums,
                                        max_psnr_val=max_psnr_val_postcard199, Dname='postcard199',SAVE_test_Results =True)
            max_psnr_val_soild200 = test(RRNet=NAFNet, RDNet =RDNet,  
                                        eval_loader=eval_loader_soild200, epoch=epoch,iters=iter_nums,
                                        max_psnr_val=max_psnr_val_soild200, Dname='soild200',SAVE_test_Results =True)


            # max_psnr_val_SIR = test(RRNet=NAFNet, RDNet =RDNet,FDNet=FDNet,  
            #                             eval_loader=eval_loader_SIR, epoch=epoch,iters=iter_nums,
            #                             max_psnr_val=max_psnr_val_SIR, Dname='SIR',SAVE_test_Results =True)
