import argparse
import logging
import os
import re
import random
import shutil
import sys
import time
from xml.etree.ElementInclude import default_loader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torch.distributions import Categorical
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (
    BaseDataSets,
    RandomGenerator,
    TwoStreamBatchSampler,
    WeakStrongAugment,
)
from dataloaders.acdc_ex import ACDCDataset
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, util
# from val_2D_MRliver import test_single_volume
from model_sam.model_dict import get_model
from utils.loss_functions.sam_loss import Mask_DC_and_BCE_loss_ap

## Medsam
from model_sam.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from networks.tiny_vit_sam import TinyViT
from networks.medsam import MedSAM_Lite, load_medsam, sam_set_uni
from utils.get_prompts import get_bbox256_cv, get_bbox256_torch

LABELED_NUM_USED = "3"
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="./data/ACDC", help="Name of Experiment")
parser.add_argument("--save_path", type=str, default="./check", help="Name of Experiment")
parser.add_argument("--exp", type=str, default="ACDC/Unimatch_medsam_fix_debug_server", help="experiment_name")
parser.add_argument("--iter_num", type=int, default=30000, help="when train sam after matchnet")
parser.add_argument("--cuda_num", type=str, default="cuda:0", help="experiment_name")
parser.add_argument("--bbox_shift", type=int, default=5, help="bbox shift")
parser.add_argument("--labeled_num", type=str, default=LABELED_NUM_USED, help="labeled data")
parser.add_argument('--labeled_id_path', type=str,default=str(LABELED_NUM_USED)+"/labeled.txt", required=False)
parser.add_argument('--unlabeled_id_path', type=str, default=str(LABELED_NUM_USED)+"/unlabeled.txt", required=False)
parser.add_argument('--ckpt_sam', type=str, 
                    default='/data/maia/gpxu/proj1/samatch/checkpoint/ACDC/MedSam_ft_'+str(LABELED_NUM_USED)+'_labeled_bs4/medsam/medsam_lite_best.pth', 
                    help='Fine-tuned SAM')
parser.add_argument('--ckpt_uni', type=str, 
                    default='/data/maia/gpxu/proj1/samatch/checkpoint/ACDC/Unimatch_'+str(LABELED_NUM_USED)+'_labeled_bs4/unet_drop/unet_drop_best_model.pth', 
                    help='Fine-tuned SAM')
parser.add_argument("--conf_thresh", type=float,default=0.95, help="confidence threshold for using pseudo-labels")
parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
parser.add_argument("--model", type=str, default="unet_drop", help="model_name")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=1e-2, help="segmentation network learning rate")
parser.add_argument("--base_lr_sam", type=float, default=5e-4, help="segmentation network learning rate")
parser.add_argument("--patch_size", type=int, default=256, help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4, help="output channel of network")
parser.add_argument("--load", default=False, action="store_true", help="restore previous checkpoint")

# cost
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")

args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"0_5" : 16, "1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    elif "MRliver" in dataset:
        ref_dict = {"1": 64, "3": 192, "5": 320, "7": 448, "9": 576, "30": 1920}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

######################
def create_model(ema=False):
    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def load_match(ckpt_match, model):
    load_weight = torch.load(ckpt_match, weights_only=False, map_location=args.cuda_num)["model"]
    model.load_state_dict(load_weight)
    return model

######################
def resize_pred(pred, size=(128, 128), mode = InterpolationMode.NEAREST):
    pred = TF.resize(pred, size, mode)
    return pred

#####################################
def train(args, snapshot_path):

    ## para setting
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    iter_num = 0
    start_epoch = 0

    # create unimatch
    model = create_model()
    model = load_match(args.ckpt_uni, model)
    # set to train
    model.to(args.cuda_num)
    model.train()
    # set optim, loss
    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ## loss function
    # for unimatch and sam
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss_U(num_classes)
    dice_loss_sam = losses.DiceLoss_U(n_classes=2)

    # load the pretrained sam model
    sam = load_medsam(args.ckpt_sam, args.cuda_num)
    sam.to(args.cuda_num)
    sam.train()
    # load optim,loss, lr
    # optimizer_sam, lr_scheduler_sam = sam_set_uni(sam, lr=args.base_lr_sam)
    optimizer_sam = optim.SGD(sam.parameters(), lr=args.base_lr_sam, momentum=0.9, weight_decay=0.0001) #base_lr_sam: 5e-5


    # prepare dataloader
    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([WeakStrongAugment(args.patch_size)]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

    # load data
    trainset_u = ACDCDataset("ACDC", args.root_path, 'train_u',
                             args.patch_size, args.unlabeled_id_path)
    trainset_l = ACDCDataset("ACDC", args.root_path, 'train_l',
                             args.patch_size, args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = ACDCDataset("ACDC", args.root_path, 'val')

    trainsampler_l = torch.utils.data.RandomSampler(trainset_l, replacement=False, num_samples=None)
    trainloader_l = DataLoader(trainset_l, batch_size=args.batch_size,
                               pin_memory=True, num_workers=0, drop_last=True, sampler=trainsampler_l)

    trainsampler_u = torch.utils.data.RandomSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=args.batch_size,
                               pin_memory=True, num_workers=0, drop_last=True, sampler=trainsampler_u)

    trainsampler_u_mix = torch.utils.data.RandomSampler(trainset_u)
    trainloader_u_mix = DataLoader(trainset_u, batch_size=args.batch_size,
                                   pin_memory=True, num_workers=0, drop_last=True, sampler=trainsampler_u_mix)

    valsampler = torch.utils.data.RandomSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=0,
                           drop_last=False, sampler=valsampler)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader_u)))

    max_epoch = max_iterations // len(trainloader_u) + 1
    previous_best = 0.0

    iter_num = int(iter_num)
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    img_temp_bbox_all = torch.zeros((batch_size, num_classes-1, 4)).to(args.cuda_num)
    # pred_w_sam = torch.zeros((batch_size, num_classes, 4)).to(args.cuda_num)
    medsam_logit_all = torch.zeros((batch_size, num_classes, 256, 256)).to(args.cuda_num)
    for epoch_num in iterator:
        total_loss = util.AverageMeter()
        total_loss_x = util.AverageMeter()
        total_loss_s = util.AverageMeter()
        total_loss_w_fp = util.AverageMeter()
        total_mask_ratio = util.AverageMeter()
        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)
        epoch_loss = [1e10 for _ in range(len(trainloader_l))]
        for i, ((img_x, mask_x),
                    (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
                    (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in enumerate(loader):
            # data load
            img_x, mask_x = img_x.to(args.cuda_num), mask_x.to(args.cuda_num)
            img_u_w = img_u_w.to(args.cuda_num)
            img_u_s1, img_u_s2 = img_u_s1.to(args.cuda_num), img_u_s2.to(args.cuda_num)
            cutmix_box1, cutmix_box2 = cutmix_box1.to(args.cuda_num), cutmix_box2.to(args.cuda_num)
            img_u_w_mix = img_u_w_mix.to(args.cuda_num)
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.to(args.cuda_num), img_u_s2_mix.to(args.cuda_num)

            ## prediction 
            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix).detach() # unlabel weakly-aug with mix-aug
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0] # prob map
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1) # pd mask from weakly-aug
            # strongly-aug: sample-1
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            # strongly-aug: sample-2
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
                
            model.train()
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0] # number of labeled and unlabeled data
            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True) # pd from normal and feature with dropout
            pred_x, pred_u_w = preds.split([num_lb, num_ulb]) # pd of labeled and unlabeld data
            pred_u_w_fp = preds_fp[num_lb:] # pd from unlabeled weakly-aug of feature-level
            # pd of stronglg-aug data
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach() 
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0] # prob map
            mask_u_w = pred_u_w.argmax(dim=1) #伪标签

            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]            

            # sam to generate psedudo-label as involves in training
            ## fine-tunning uni and sam
            img_x_3ch = torch.cat((img_x, img_x, img_x), dim=1) # 4x1xHxW
            img_u_w_3ch = torch.cat((img_u_w, img_u_w, img_u_w), dim=1)
            # img_sam = torch.cat((img_x_3ch, img_u_w_3ch),dim=0) # normalized # B3HW
            ## there are three classes, so generate three boxes. THen select an box randomly for a prmpmt
            for n in range(0, num_classes-1): # cls = 1, 2, 3
                img_temp_bbox_all[:, n, :] = get_bbox256_torch(mask_x==(n+1))[:,0,:] # Bx1x4
            # get one box from gt mask           
            idx = random.choice([0,1,2])
            img_gt_bbox = img_temp_bbox_all[:, idx, :].unsqueeze(1) # Bx1x4
            mask_x_one = (mask_x == (idx+1)).long() # notice the object class: 1, 2, 3
            # do predition, and generate a prob map of one class
            ## prediction 
            with torch.no_grad():
                sam.eval()
                medsam_logit_x, iou_pred = sam(img_x_3ch, img_gt_bbox)
            # medsam_mask_x = torch.sigmoid(medsam_logit_x) > 0.5 # 8x1xHxW
            pred_sam_pos = torch.sigmoid(medsam_logit_x) # B*1*H*W
            pred_sam_neg = 1 - pred_sam_pos
            pred_x_sam = torch.cat((pred_sam_neg, pred_sam_pos), dim=1)

            # get three boxes from match pd mask, then do prediction by sam orderly
            for n in range(1, num_classes):
                ## prediction 
                with torch.no_grad():
                    sam.eval()
                    img_pd_bbox = get_bbox256_cv(mask_u_w==n, bbox_shift=args.bbox_shift)
                    medsam_logit, iou_pred = sam(img_u_w_3ch, img_pd_bbox)
                    medsam_logit = torch.sigmoid(medsam_logit)
                    medsam_logit_all[:, n, :, :] = medsam_logit[:,0,:,:]
            sam.train()
            medsam_logit_all[:, 0, :, :] = (1 - torch.mean(medsam_logit_all[:, 1:, :, :], dim=1, keepdim=True)).squeeze(1) # for the background
            pred_w_sam = medsam_logit_all.softmax(dim=1)
            mask_u_w_sam = pred_w_sam.argmax(dim=1)

            # labeled loss from sam
            # pred_x_sam: Bx2xHxW, mask_x: BxHxW
            # pred_x_sam.requires_grad = True
            loss_x_sam = (ce_loss(pred_x_sam, mask_x_one) + 
                        dice_loss_sam(pred_x_sam, mask_x_one.unsqueeze(1).float())) / 2.0
            # unlabed loss from sam
            # pred_w_sam.requires_grad = True
            loss_u_w_sam = (ce_loss(pred_w_sam, mask_u_w) + 
                        dice_loss(pred_w_sam, mask_u_w.unsqueeze(1).float())) / 2.0
            loss_sam = loss_x_sam + loss_u_w_sam * 0.25
            loss_sam.requires_grad = True ## this is wrong!!! the sam did not train
            epoch_loss[i] = loss_sam.item()
            ## update sam model
            optimizer_sam.zero_grad()
            loss_sam.backward()
            optimizer_sam.step()

            ## loss for unimatch
            # conf_u_w_sam = resize_pred(pred_sam_pos[num_lb:,::], size=(256, 256), mode=InterpolationMode.BILINEAR)
            # conf_u_w_sam = conf_u_w_sam.squeeze(1)
            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w_sam.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w_sam.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

            # labeled loss from unimatch: pred_x: B2HW, mask_x: BHW
            loss_x = (ce_loss(pred_x, mask_x) + 
                        dice_loss(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())) / 2.0
            
            # unlabeled loss from unimatch
            loss_u_s1 = dice_loss(pred_u_s1.softmax(dim=1), 
                                    mask_u_w_cutmixed1.unsqueeze(1).float(),
                                    ignore=(conf_u_w_cutmixed1 < args.conf_thresh).float())
            
            loss_u_s2 = dice_loss(pred_u_s2.softmax(dim=1), 
                                    mask_u_w_cutmixed2.unsqueeze(1).float(),
                                    ignore=(conf_u_w_cutmixed2 < args.conf_thresh).float())
            
            loss_u_w_fp = dice_loss(pred_u_w_fp.softmax(dim=1), 
                                    mask_u_w_sam.unsqueeze(1).float(),
                                        ignore=(conf_u_w < args.conf_thresh).float())
            
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())

            mask_ratio = (conf_u_w >= args.conf_thresh).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())
            
            iters = epoch_num * len(trainloader_u) + i
            lr = args.base_lr * (1 - iters / max_iterations) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1                

            # record
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
            writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            logging.info("iteration %d : model loss : %f" % (iter_num, loss.item()))

            if (i % (len(trainloader_u) // 8) == 0):
                logging.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, 
                                            total_loss_w_fp.avg, total_mask_ratio.avg))
        
        # # complete an epoch, update lr_sam
        # epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
        # lr_scheduler_sam.step(epoch_loss_reduced)

        # do evaluation
        dice_class = [0] * 3
        with torch.no_grad():
            model.eval()
            for img, mask in valloader:
                img, mask = img.to(args.cuda_num), mask.to(args.cuda_num) # BHWC
                h, w = img.shape[-2:]
                img = F.interpolate(img, (args.patch_size, args.patch_size), mode='bilinear', align_corners=False)
 
                img = img.permute(1, 0, 2, 3)
                pred = model(img)
                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
                pred = pred.argmax(dim=1).unsqueeze(0)
 
                for cls in range(1, args.num_classes):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls-1] += 2.0 * inter / union

        dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)

        for (cls_idx, dice) in enumerate(dice_class):
                logging.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, 'acdc', dice))
        logging.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))
           
        writer.add_scalar('eval/MeanDice', mean_dice, iter_num)
        CLASSES = {'acdc': ['Right Ventricle', 'Myocardium', 'Left Ventricle']}
        for i, dice in enumerate(dice_class):
                writer.add_scalar('eval/%s_dice' % (CLASSES['acdc'][i]), dice, iter_num)


        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': iter_num,
                'previous_best': previous_best,
            }
        
        torch.save(checkpoint, os.path.join(snapshot_path, 'match_latest.pth'))
        ## save sam
        save_best_sam = os.path.join(snapshot_path, "medsam_lite_latest.pth")
        torch.save(sam.state_dict(), save_best_sam, _use_new_zipfile_serialization=False)
        if is_best:
                torch.save(checkpoint, os.path.join(snapshot_path, args.model + '_best_model.pth'))
                ## save sam
                save_best_sam = os.path.join(snapshot_path, "medsam_lite_best.pth")
                torch.save(sam.state_dict(), save_best_sam, _use_new_zipfile_serialization=False)
                if mean_dice > 0.1: 
                    torch.save(checkpoint, os.path.join(snapshot_path, 
                                    "iter_"+str(iter_num)+"_dice_"+str(round(mean_dice,3))+'.pth'))


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./checkpoint/{}_{}_labeled_bs{}/{}".format(
        args.exp, args.labeled_num, args.batch_size, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)