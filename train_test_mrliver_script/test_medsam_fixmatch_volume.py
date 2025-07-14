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
from networks.medsam import load_medsam
from utils.get_prompts import get_bbox256_cv, get_bbox256

#  =========================================== parameters setting ==================================================
MODEL_PATH = "FixMatch_medsam_3_labeled_bs4_shift_box10"
parser = argparse.ArgumentParser(description='Networks')
parser.add_argument('--root_path', type=str, default='/home/gxu/proj1/smatch/checkpoint/MRliver', help='the path of MatchNet')
parser.add_argument('--data_path', type=str, default='/home/gxu/proj1/smatch/data/MRliver', help='the path of MatchNet')
parser.add_argument('--matchnet', type=str, default='/home/gxu/proj1/smatch/checkpoint/MRliver/'+MODEL_PATH+'/unet/unet_best_model.pth', help='the path of MatchNet')
parser.add_argument('--sam_path', type=str, default='/home/gxu/proj1/smatch/checkpoint/MRliver/'+MODEL_PATH+'/unet/medsam/medsam_lite_best.pth', help='the path of MatchNet')
parser.add_argument('--save_path', type=str, default='/home/gxu/proj1/smatch/checkpoint/MRliver/'+MODEL_PATH, help='the path of MatchNet')

parser.add_argument('--modelname', default='MedSAM', type=str, help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
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

parser.add_argument('--cuda_num', type=str, default="cuda:0", help='cuda index')


args = parser.parse_args()


def main():
    print("task", args.task, "SAM checkpoints:", args.sam_path)
    device = args.cuda_num

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
    
    # load the pretrained sam model
    model = load_medsam(args.sam_path, args.cuda_num)
    model.eval()
#  ========================================================================= begin to evaluate the model ============================================================================

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    # input = torch.randn(1, 3, args.encoder_input_size, args.encoder_input_size).cuda()
    # points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    # flops, params = profile(model, inputs=(input, points), )
    # print('Gflops:', flops/1000000000, 'params:', params)


    # load matchnet
    matchnet = load_matchnet(args.matchnet)
    matchnet = matchnet.to(args.cuda_num)
    matchnet.eval()
    ########## create filefolder
    save_result_path = os.path.join(args.save_path, "sam_match_pd")
    if os.path.exists(save_result_path):
        shutil.rmtree(save_result_path)
    os.makedirs(save_result_path)

    ########### evaluate each patient
    with open(args.data_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    first_total = 0.0
    for case in tqdm(image_list):
        first_metric = test_single_volume(
            case, model, matchnet, save_result_path, args)
        first_total += np.asarray(first_metric)
        print("case: {}, metric: {}".format(case, first_metric))
    avg_metric = first_total / len(image_list)
    return avg_metric

###########################: load fixmatch: student model
def load_matchnet(net_path):
    # build model
    matchnet = net_factory(net_type="unet", in_chns=1, class_num=2)
    # load trained parameters
    matchnet.load_state_dict(torch.load(net_path, map_location='cpu', weights_only=True)["state_dict"])

    return matchnet

#############################: test each volume
def test_single_volume(case, net, matchnet, test_save_path, args):
    h5f = h5py.File(args.data_path + "/volumes/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    image = (image - image.min()) / (image.max()- image.min()+1e-9) # Notice: normalize
    
    net.eval()
    prediction = np.zeros_like(label)
    for ind in range(0, image.shape[2]):
        slice = image[:, :, ind]
        # mask = label[:,:,ind]
        # obtain label from matchnet
        img = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(args.cuda_num)
        # do prediction
        out_main = matchnet(img) # B2HW: the number 2 is class number
        out_prob = torch.softmax(out_main, dim=1)
        pseudo_outputs = torch.argmax(out_prob.detach(), dim=1, keepdim=False)
        bbox_prmpt = get_bbox256_cv(pseudo_outputs, bbox_shift=3)
        input = torch.cat((img, img, img), dim=1) ## this step is very important
        # input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()    
        with torch.no_grad():
            medsam_logit, iou_pred = net(input, bbox_prmpt)
            medsam_mask = torch.sigmoid(medsam_logit) > 0.5
            pred = medsam_mask.detach().squeeze(0).squeeze(0).cpu().numpy()  # (b, c, h, w)
            pred = pred.astype("int") 
            # out = torch.argmax(torch.softmax(
            #     out_main['masks'], dim=1), dim=1).squeeze(0)
            # out = out.cpu().detach().numpy()

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
    with open(os.path.join(args.save_path, "metric_medsamLit_box_shift3_pd.txt"),'w') as f:
        res = "%.4f\t%.3f\t%.3f" % (avg_metric[0], avg_metric[1], avg_metric[2])
        f.write(res)