import argparse
import logging
import os
import re
import random
import shutil
import sys
import time
from xml.etree.ElementInclude import default_loader

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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
from val_2D import test_single_volume

LABELED_NUM = "0_5" # 0_5, 1, 3
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="./data/ACDC", help="Name of Experiment")
parser.add_argument("--save_path", type=str, default="./check", help="Name of Experiment")
parser.add_argument("--exp", type=str, default="ACDC/Unimatch", help="experiment_name")
parser.add_argument("--cuda_num", type=str, default="cuda:0", help="experiment_name")
parser.add_argument("--model", type=str, default="unet_drop", help="model_name")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--patch_size", type=int, default=256, help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4, help="output channel of network")
parser.add_argument("--load", default=False, action="store_true", help="restore previous checkpoint")
parser.add_argument( "--conf_thresh", type=float, default=0.95, help="confidence threshold for using pseudo-labels",)
parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
parser.add_argument("--labeled_num", type=str, default=LABELED_NUM, help="labeled data")
parser.add_argument('--labeled_id_path', type=str,default=str(LABELED_NUM) + "/labeled.txt", required=False)
parser.add_argument('--unlabeled_id_path', type=str, default=str(LABELED_NUM)+"/unlabeled.txt", required=False)
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")


args = parser.parse_args()
def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"0_5":16, "1": 32, "3": 68, "7": 136,
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


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


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
    # labeled_idxs = list(range(0, labeled_slice))
    # unlabeled_idxs = list(range(labeled_slice, total_slices))

    model = create_model()

    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

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
    # set to train
    model.to(args.cuda_num)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss_U(num_classes)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader_u)))

    max_epoch = max_iterations // len(trainloader_u) + 1
    previous_best = 0.0

    iter_num = int(iter_num)

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:
        total_loss = util.AverageMeter()
        total_loss_x = util.AverageMeter()
        total_loss_s = util.AverageMeter()
        total_loss_w_fp = util.AverageMeter()
        total_mask_ratio = util.AverageMeter()
        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)
        for i, ((img_x, mask_x),
                    (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
                    (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in enumerate(loader):
            img_x, mask_x = img_x.to(args.cuda_num), mask_x.to(args.cuda_num)
            img_u_w = img_u_w.to(args.cuda_num)
            img_u_s1, img_u_s2 = img_u_s1.to(args.cuda_num), img_u_s2.to(args.cuda_num)
            cutmix_box1, cutmix_box2 = cutmix_box1.to(args.cuda_num), cutmix_box2.to(args.cuda_num)
            img_u_w_mix = img_u_w_mix.to(args.cuda_num)
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.to(args.cuda_num), img_u_s2_mix.to(args.cuda_num)

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
                
            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach() 
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1) #伪标签

            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
           

            loss_x = (ce_loss(pred_x, mask_x) + dice_loss(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())) / 2.0

            loss_u_s1 = dice_loss(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed1 < args.conf_thresh).float())
            
            loss_u_s2 = dice_loss(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed2 < args.conf_thresh).float())
            
            loss_u_w_fp = dice_loss(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1).float(),
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

            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
            writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            logging.info("iteration %d : model loss : %f" % (iter_num, loss.item()))

            if (i % (len(trainloader_u) // 8) == 0):
                logging.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, 
                                            total_loss_w_fp.avg, total_mask_ratio.avg))
        
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
        
        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))
        if is_best:
                torch.save(checkpoint, os.path.join(snapshot_path,args.model + '_best_model.pth'))
                if mean_dice > 0.1: 
                    torch.save(checkpoint, os.path.join(snapshot_path, "model_iter_" + str(iter_num) + "_dice_"+ str(round(mean_dice,2))+'.pth'))

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