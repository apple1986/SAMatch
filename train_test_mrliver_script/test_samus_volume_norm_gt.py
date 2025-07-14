import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import shutil
import argparse
from random import randint
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
import random
from utils.config import get_config
from utils.evaluation_ap import get_eval

from model_sam.model_dict import get_model
from utils.data_mrliver import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from thop import profile
import h5py
from scipy.ndimage import zoom
from medpy import metric
from tqdm import tqdm
from torchvision.transforms import functional as F

import SimpleITK as sitk
from networks.net_factory import net_factory
## set on config
#  =========================================== parameters setting ==================================================
parser = argparse.ArgumentParser(description='Networks')
parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS') 
parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS') 
parser.add_argument('--task', default='MRliver', help='task or dataset name')
parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
parser.add_argument('--ckpt', type=str, default='/home/gxu/proj1/smatch/checkpoint/SAM/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # 8 # SAMed is 12 bs with 2n_gpu and lr is 0.005
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') # True
parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')

args = parser.parse_args()
opt = get_config(args.task)  # please configure your hyper-parameter

def main():
    print("task", args.task, "checkpoints:", opt.load_path)
    opt.mode = "val"
    #opt.classes=2
    opt.visual = False
    #opt.eval_mode = "patient"
    opt.modelname = args.modelname
    device = torch.device(opt.device)

     #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 300 # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    # load sam
    checkpoint = torch.load(opt.load_path, map_location='cuda:0', weights_only=True)
    #------when the load model is saved under multiple GPU
    new_state_dict = {}
    for k,v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
#  ========================================================================= begin to evaluate the model ============================================================================

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()
    points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    flops, params = profile(model, inputs=(input, points), )
    print('Gflops:', flops/1000000000, 'params:', params)

    model.eval()
    ########## create filefolder
    if os.path.exists(opt.result_path):
        shutil.rmtree(opt.result_path)
    os.makedirs(opt.result_path)

    ########### evaluate each patient
    with open(opt.data_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    first_total = 0.0
    for case in tqdm(image_list):
        first_metric = test_single_volume(
            case, model, opt.result_path, opt)
        first_total += np.asarray(first_metric)
        print("case: {}, metric: {}".format(case, first_metric))
    avg_metric = first_total / len(image_list)
    return avg_metric

############################: load point for pseudo-label
def generate_unique_random_numbers(n, start, end):
    # generate n number in [start, end]
    return random.sample(range(start, end + 1), n)

def find_point_label_pseudo(pd_sam):
    # input: pd_sam is a probability map
    # get 10 points for pseudo-label: 1 for highest probability, 9 for bg
    B, _, _ = pd_sam.shape
    points_coord = torch.zeros((B, 10, 2), device=pd_sam.device)
    points_label = torch.zeros((B, 10), device=pd_sam.device)
    for n in range(B):
        pd_one = pd_sam[n,:,:] # HW
        idx_fg = torch.argwhere(pd_one > 0.95).to(device=pd_sam.device) # 1 is the class label, 0.5 is threshold
        idx_bg = torch.argwhere(pd_one < 0.05).to(device=pd_sam.device) 
        if len(idx_fg) == 0:
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            ## sample 10 points in a random way
            random_numbers = generate_unique_random_numbers(10, 0, len(idx_bg)-1)
            points_coord[n,:,:] = idx_bg[random_numbers] # 10x2
            # points_label[0,:] = torch.zeros((len(random_numbers)), device=gt_sam.device) # 1x10
        else:
            idx_fg[:, [0,1]] = idx_fg[:, [1,0]] # make [row, col] to [x, y]
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            # five points from object, five points for background
            random_numbers = generate_unique_random_numbers(1, 0, len(idx_fg)-1)
            # foreground: points and labels
            points_coord[n, :1, :] = idx_fg[random_numbers]
            points_label[n, :1] = 1
            # backgrouond: points and labels
            random_numbers = generate_unique_random_numbers(9, 0, len(idx_bg)-1)
            points_coord[n, 1:, :] = idx_bg[random_numbers]
            points_label[n, 1:] = 0
    return points_coord, points_label

def find_point_label_pseudo_prob(pd_sam):
    # input: pd_sam is a probability map
    # get 10 points for pseudo-label: 1 for highest probability, 9 for bg
    B, _, _ = pd_sam.shape
    points_coord = torch.zeros((B, 10, 2), device=pd_sam.device)
    points_label = torch.zeros((B, 10), device=pd_sam.device)
    for n in range(B):
        pd_one = pd_sam[n,:,:] # HW
        # idx_fg = torch.argwhere(pd_one > 0.5).to(device=pd_sam.device) # 1 is the class label, 0.5 is threshold
        idx_fg = torch.argwhere(pd_one == pd_one.max()).to(device=pd_sam.device)
        idx_bg = torch.argwhere(pd_one < 0.8).to(device=pd_sam.device) 
        if len(idx_fg) == 0:
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            ## sample 10 points in a random way
            random_numbers = generate_unique_random_numbers(10, 0, len(idx_bg)-1)
            points_coord[n,:,:] = idx_bg[random_numbers] # 10x2
            # points_label[0,:] = torch.zeros((len(random_numbers)), device=gt_sam.device) # 1x10
        else:
            idx_fg[:, [0,1]] = idx_fg[:, [1,0]] # make [row, col] to [x, y]
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            # five points from object, five points for background
            random_numbers = generate_unique_random_numbers(1, 0, len(idx_fg)-1)
            # foreground: points and labels
            points_coord[n, :1, :] = idx_fg[random_numbers]
            points_label[n, :1] = 1
            # backgrouond: points and labels
            random_numbers = generate_unique_random_numbers(9, 0, len(idx_bg)-1)
            points_coord[n, 1:, :] = idx_bg[random_numbers]
            points_label[n, 1:] = 0
    return points_coord, points_label


############################: find point
def fixed_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[len(indices)//2]

    return pt[np.newaxis, :], point_label

###########################: get box
def fixed_bbox(mask, class_id = 1, img_size=256):
    indices = np.argwhere(mask == class_id) # Y X (0, 1)
    indices[:, [0,1]] = indices[:, [1,0]]
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])
    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])
    return np.array([minx, miny, maxx, maxy])


#############################: test each volume
def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.data_path + "/volumes/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    
    net.eval()
    prediction = np.zeros_like(label)
    for ind in range(0, image.shape[2]):
        slice = image[:, :, ind]
        mask = label[:,:,ind]
        slice = (slice - slice.min()) / (slice.max()- slice.min()+1e-9) # norm
        input_point, input_label = fixed_click(np.array(mask), 1)  # fixed_click
        input_point = torch.as_tensor(input_point, dtype=torch.float32, device=net.device).unsqueeze(0)
        input_label = torch.as_tensor(input_label, dtype=torch.int, device=net.device).unsqueeze(0).unsqueeze(0)
        point_label = (input_point, input_label)
        # bbox = fixed_bbox(np.array(mask), 1, 256) # fixed_bbox
        # bbox = torch.as_tensor(bbox, dtype=torch.int, device=net.device).unsqueeze(0)
        bbox = None
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        # input = F.to_tensor(slice.astype(np.uint8)).unsqueeze(0).cuda() ## this step is very important
        input = F.to_tensor(slice).unsqueeze(0).float().cuda() ## this step is very important
        # input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()    
        with torch.no_grad():
            out_main = net(input, point_label, bbox) # 
            predict = torch.sigmoid(out_main['masks'])
            predict = predict.detach().cpu().numpy()  # (b, c, h, w)
            out = predict[0, 0, :, :] > 0.5  # (b, h, w)  
            out = out.astype("int") 
            # out = torch.argmax(torch.softmax(
            #     out_main['masks'], dim=1), dim=1).squeeze(0)
            # out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[:, :, ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)

    ## save results
    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    # sitk.WriteImage(img_itk, os.path.join(test_save_path, case +"_img.nii.gz"))
    # sitk.WriteImage(lab_itk, os.path.join(test_save_path, case +"_gt.nii.gz"))
    return first_metric

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    
    return dice, hd95, asd


if __name__ == '__main__':
    avg_metric = main()
    print("%.4f %.3f %.3f" % (avg_metric[0], avg_metric[1], avg_metric[2]))
    with open(os.path.join(opt.save_path, "eval_smatch_sam.txt"),'w') as f:
        res = "%.4f\t%.3f\t%.3f" % (avg_metric[0], avg_metric[1], avg_metric[2])
        f.write(res)