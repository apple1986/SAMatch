## use original SAM to generate pseudo label

from segment_anything import SamPredictor, sam_model_registry
import argparse
import logging
import os
import re
import random
import shutil
import sys
import time
import matplotlib
from sympy import im
matplotlib.use('TkAgg')
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
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
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset_norm import (
    BaseDataSets,
    RandomGenerator,
    TwoStreamBatchSampler,
    WeakStrongAugment,
)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, util
from val_2D_MRliver import test_single_volume
### medsam
from model_sam.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from networks.tiny_vit_sam import TinyViT
from networks.medsam import MedSAM_Lite, load_medsam, sam_set
from utils.get_prompts import get_bbox256_cv

################# for sam
from model_sam.model_dict import get_model
from utils.loss_functions.sam_loss import Mask_DC_and_BCE_loss_ap

import cv2

#################

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="./data/MRliver", help="Name of Experiment")
parser.add_argument("--save_path", type=str, default="./checkpoint", help="Name of Experiment")
parser.add_argument("--exp", type=str, default="MRliver/FixMatch_medsam", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="model_name") 
parser.add_argument("--sam_model", type=str, default="medsam", help="model_name") 
parser.add_argument("--iter_num", type=int, default=3000, help="when train sam after matchnet")
## select gpu: setting the net_factory and test_single_volume function
parser.add_argument('--cuda_num', type=str,  default="cuda:0", help='random seed')
parser.add_argument( "--conf_thresh", type=float,default=0.95, help="confidence threshold for using pseudo-labels")
parser.add_argument("--max_iterations", type=int, default=60000, help="maximum epoch number to train")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--base_lr_sam", type=float, default=5e-5, help="segmentation network learning rate")
parser.add_argument("--labeled_num", type=int, default=5, 
                    help="how many labeled cases are used. Note that each case includes some slices")
parser.add_argument("--labeled_bs", type=int, default=2, 
                    help="labeled_batch_size per gpu, how may labeded slice used for training")
parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
parser.add_argument("--bbox_shift", type=int, default=10, help="use to enlarge bbox")

parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=2, help="output channel of network")
parser.add_argument("--load", default=False, action="store_true", help="restore previous checkpoint")

# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")
# for sam
parser.add_argument('--ckpt', type=str, 
                    default='/home/gxu/proj1/smatch/checkpoint/SAM/sam_vit_b_01ec64.pth', 
                    help='Pretrained checkpoint of original SAM')
parser.add_argument('--ckpt_sam', type=str, 
                    default='/home/gxu/proj1/smatch/checkpoint/SAM/lite_medsam.pth', 
                    help='Fine-tuned SAM')
# parser.add_argument('--ckpt_sam', type=str, 
#                     default='/home/gxu/proj1/smatch/checkpoint/MRliver/FixMatch_medsam_5_labeled_bs8_shift_box50/medsam/medsam_lite_best.pth', 
#                     help='Fine-tuned SAM')
# parser.add_argument('--ckpt_matchnet', type=str, 
#                     default='/home/gxu/proj1/smatch/checkpoint/MRliver/FixMatch_medsam_5_labeled_bs8_shift_box50/unet/unet_best_model.pth', 
#                     help='Fine-tuned SAM')
args = parser.parse_args()

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    elif "MRliver" in dataset:
        ref_dict = {"1": 64, "3": 192, "5": 320, "7": 448, "9": 576, "11": 704, "30": 1920}
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


######################train setting
# cread match
def create_model(ema=False):
    model = net_factory(net_type=args.model, in_chns=1, 
                        class_num=args.num_classes, cuda_num=args.cuda_num)
    if ema:
        for param in model.parameters():
            param.detach_()

    # ## load pretrained model
    # model.load_state_dict(torch.load(args.ckpt_matchnet, weights_only=True)["state_dict"])

    return model

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def get_comp_loss(weak, strong, ce_loss):
    """get complementary loss and adaptive sample weight.
    Compares least likely prediction (from strong augment) with argmin of weak augment.

    Args:
        weak (batch): weakly augmented batch
        strong (batch): strongly augmented batch

    Returns:
        comp_loss, as_weight
    """
    il_output = torch.reshape(
        strong,
        (
            args.batch_size,
            args.num_classes,
            args.patch_size[0] * args.patch_size[1],
        ),
    )
    # calculate entropy for image-level preds (tensor of length labeled_bs)
    as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(args.patch_size[0] * args.patch_size[1]))
    # batch level average of entropy
    as_weight = torch.mean(as_weight)
    # complementary loss
    comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
    comp_loss = as_weight * ce_loss(
        torch.add(torch.negative(strong), 1),
        comp_labels,
    )
    return comp_loss, as_weight
def normalize(tensor):
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    result = tensor - min_val
    result = result / max_val
    return result

def resize_pred(pred, mode = InterpolationMode.NEAREST):
    # resiez features
    # pred = F.interpolate(pred, (128, 128), mode=mode)
    pred = TF.resize(pred, (128, 128), mode)
    return pred


#######################matchnet setting
def match_set(matchnet, base_lr, num_classes):
    optimizer = optim.SGD(matchnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) # base_lr=1e-2
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    return optimizer, ce_loss, dice_loss

def cal_iou(result, reference):
    
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    
    iou = intersection.float() / (union.float()+1e-9)
    
    return iou.unsqueeze(1)
##############################SAM training
def sam_train_sup(img_w_lb_sam, img_w_bbox, img_w_gt, sam, dice_loss_sam, ce_loss_sam, iou_loss_sam):
    ####################################: for medsam
    # finetune medsam with labeled data
    medsam_logit, iou_pred = sam(img_w_lb_sam, img_w_bbox)
    medsam_mask = torch.sigmoid(medsam_logit) > 0.5
    medsam_logit = medsam_logit.squeeze(1)

    # cal loss
    loss_seg_w_sam = dice_loss_sam(medsam_logit, img_w_gt)
    loss_ce_w_sam = ce_loss_sam(medsam_logit, img_w_gt.float())
    loss_mask_w_sam = loss_seg_w_sam + loss_ce_w_sam
    iou_gt = cal_iou(medsam_mask, img_w_gt.bool())
    loss_iou_w_sam = iou_loss_sam(iou_pred, iou_gt)
    total_loss_w_sam = loss_mask_w_sam * 2 + loss_iou_w_sam
    # print('loss_mask_w_sam: {}, loss_iou_w_sam: {}'.format(loss_mask_w_sam, loss_iou_w_sam))
    return total_loss_w_sam

def matchnet_train(args, weak_batch, strong_batch, label_batch, matchnet, ce_loss, dice_loss, consistency_weight):
    # outputs from matchnet
    outputs_weak = matchnet(weak_batch) # without normalize
    outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
    outputs_strong = matchnet(strong_batch)
    outputs_strong_soft = torch.softmax(outputs_strong, dim=1)

    # minmax normalization for softmax outputs before applying mask
    pseudo_mask = (normalize(outputs_weak_soft) > args.conf_thresh).float()
    outputs_weak_masked = outputs_weak_soft * pseudo_mask
    pseudo_outputs = torch.argmax(outputs_weak_masked[args.labeled_bs :].detach(), dim=1, keepdim=False)
    # update student model only
    # supervised loss
    sup_loss = ce_loss(
        outputs_weak[: args.labeled_bs], label_batch[: args.labeled_bs].long(),) + \
        dice_loss(outputs_weak_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1)
                    )

    # complementary loss and adaptive sample weight for negative learning
    comp_loss, as_weight = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft, ce_loss=ce_loss)
    # unsupervised loss # ignore=(pseudo_outputs < args.conf_thresh).float())
    unsup_loss = (
        ce_loss(outputs_strong[args.labeled_bs :], pseudo_outputs)
        + dice_loss(outputs_strong_soft[args.labeled_bs :], pseudo_outputs.unsqueeze(1))
        + as_weight * comp_loss
    )
    ## get the total loass
    loss_stu = sup_loss + consistency_weight * unsup_loss

    return loss_stu, outputs_weak

def write_log_match(args, epoch_num, iter_num, best_performance, base_lr, max_iterations, optimizer, writer, consistency_weight, weak_batch, label_batch, outputs_weak, valloader,db_val, matchnet, sam, loss_stu):
    # hyper-parameter, data, model ,loss
    # update learning rate
    lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_  
    # update iteration number and save training log
    writer.add_scalar("lr", lr_, iter_num)
    writer.add_scalar("consistency_weight/consistency_weight", consistency_weight, iter_num)
    writer.add_scalar("loss/model_loss", loss_stu, iter_num)
    logging.info("iteration %d : model loss : %f" % (iter_num, loss_stu.item()))

    # update iteration number and save training log
    writer.add_scalar("lr", lr_, iter_num)
    writer.add_scalar("consistency_weight/consistency_weight", consistency_weight, iter_num)
    writer.add_scalar("loss/model_loss", loss_stu, iter_num)
    logging.info("iteration %d : model loss : %f" % (iter_num, loss_stu.item()))
    if iter_num % 50 == 0:
        image = weak_batch[1, 0:1, :, :]
        writer.add_image("train/Image", image, iter_num)
        outputs_weak = torch.argmax(torch.softmax(outputs_weak, dim=1), dim=1, keepdim=True)
        writer.add_image("train/model_Prediction", outputs_weak[1, ...] * 50, iter_num)

        labs = label_batch[1, ...].unsqueeze(0) * 50
        writer.add_image("train/GroundTruth", labs, iter_num)

    if iter_num > 0 and iter_num % 200 == 0:
        matchnet.eval()
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(valloader):
            metric_i = test_single_volume(
                sampled_batch["image"],
                sampled_batch["label"],
                matchnet,
                classes=args.num_classes,
                cuda_num=args.cuda_num,
            )
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(db_val) # len(db_val) = 6; total 6 cases

        mean_dice = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        writer.add_scalar("info/model_val_mean_dice", mean_dice, iter_num)
        writer.add_scalar("info/model_val_mean_hd95", mean_hd95, iter_num)

        if mean_dice > best_performance:
            best_performance = mean_dice
            save_mode_path = os.path.join(
                snapshot_match_path,
                "model_iter_{}_dice_{}.pth".format(iter_num, round(best_performance, 4)),
            )
            save_best = os.path.join(snapshot_match_path, "{}_best_model.pth".format(args.model))
            util.save_checkpoint(epoch_num, matchnet, optimizer, loss_stu, save_mode_path)
            util.save_checkpoint(epoch_num, matchnet, optimizer, loss_stu, save_best)
            ## save sam
            save_best_sam = os.path.join(snapshot_match_path, "best_sam_model.pth")
            torch.save(sam.state_dict(), save_best_sam, _use_new_zipfile_serialization=False)

        logging.info("iteration %d : model_mean_dice : %f model_mean_hd95 : %f" % (iter_num, mean_dice, mean_hd95))
        matchnet.train()

    if iter_num % 3000 == 0:
        save_mode_path = os.path.join(snapshot_match_path, "model_iter_" + str(iter_num) + ".pth")
        util.save_checkpoint(epoch_num, matchnet, optimizer, loss_stu, save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
    return best_performance

def write_log_sam(epoch_num, epoch_loss, best_loss, sam, lr_scheduler_sam, optimizer_sam, snapshot_sam_path):
    epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)            
    lr_scheduler_sam.step(epoch_loss_reduced)
    model_weights = sam.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch_num,
        "optimizer": optimizer_sam.state_dict(),
        "loss": epoch_loss_reduced,
    }
    torch.save(checkpoint, os.path.join(snapshot_sam_path, "medsam_lite_latest.pth"))
    if epoch_loss_reduced < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
        best_loss = epoch_loss_reduced
        checkpoint["best_loss"] = best_loss
        torch.save(checkpoint, os.path.join(snapshot_sam_path, "medsam_lite_best.pth"))
    return best_loss

def matchnet_output(args, weak_batch, strong_batch, matchnet, threshold=0.5):
    # outputs for model
    outputs_weak = matchnet(weak_batch) # without normalize
    outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
    outputs_strong = matchnet(strong_batch)
    outputs_strong_soft = torch.softmax(outputs_strong, dim=1)

    # minmax normalization for softmax outputs before applying mask
    pseudo_mask = (normalize(outputs_weak_soft) > threshold).float()
    # pseudo_mask = (normalize(outputs_weak_soft) > 0.5).float()
    outputs_weak_masked = outputs_weak_soft * pseudo_mask
    pseudo_outputs = torch.argmax(outputs_weak_masked[args.labeled_bs :].detach(), dim=1, keepdim=False)

    return outputs_strong, outputs_strong_soft,  outputs_weak, outputs_weak_soft, pseudo_outputs


###################################
def train(args, snapshot_match_path, snapshot_sam_path):
    ## 1. hyper-para setting
    base_lr = args.base_lr
    lr_sam = args.base_lr_sam
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    cuda_num = args.cuda_num
    iter_num = 0 # number of iterations
    start_epoch = 0 # number of epoches

    ## 2. data setting 
    # prepare data
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
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)
    # dataloader
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=1,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # 3. model setting 
    # load the pretrained samus model
    sam = load_medsam(args.ckpt_sam, args.cuda_num)
    # load optim,loss, lr
    optimizer_sam, lr_scheduler_sam, dice_loss_sam, ce_loss_sam, iou_loss_sam = sam_set(sam, lr=lr_sam)

    # load match model
    matchnet = create_model()
    # load optim, loss
    optimizer, ce_loss, dice_loss = match_set(matchnet, base_lr, num_classes)

    # set to train
    matchnet.train()
    sam.train()
    # write record
    writer = SummaryWriter(snapshot_match_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    # cal training iteration numbers
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    print("\nstart_epoch:{},  max_epoch: {}".format(start_epoch, max_epoch))
    best_loss = 1e10
    for epoch_num in iterator:
        print(epoch_num)
        # train sam and match separately
        if epoch_num < (max_epoch // 2):
            epoch_loss = [1e10 for _ in range(len(trainloader))]
            # load training samples
            for step, sampled_batch in enumerate(trainloader):
                iter_num = iter_num + 1
                # load data
                weak_batch, strong_batch, label_batch, label_bbox = (
                    sampled_batch["image_weak"],
                    sampled_batch["image_strong"],
                    sampled_batch["label"], 
                    sampled_batch["bbox"],
                )
                weak_batch, strong_batch, label_batch, label_bbox = (
                    weak_batch.to(cuda_num),
                    strong_batch.to(cuda_num),
                    label_batch.to(cuda_num),
                    label_bbox.to(cuda_num)
                )
                
                ####################################: for medsam training
                # prepare labeled data
                img_w_lb = weak_batch[: args.labeled_bs, ::] # Bx1x256x256
                img_w_lb_sam = torch.cat((img_w_lb, img_w_lb, img_w_lb), dim=1) # Bx3x256x256
                img_w_gt = label_batch[: args.labeled_bs, ::] # Bx256x256
                #  box from gt
                img_w_bbox = label_bbox[: args.labeled_bs, ::].unsqueeze(1)
                # get total loss
                total_loss_w_sam = sam_train_sup(img_w_lb_sam, img_w_bbox, img_w_gt, sam, dice_loss_sam, ce_loss_sam, iou_loss_sam)
                # record loss
                epoch_loss[step] = total_loss_w_sam.item()
                # update gradient
                optimizer_sam.zero_grad()
                total_loss_w_sam.backward()
                optimizer_sam.step()

                ####################################: for matchnet training
                consistency_weight = get_current_consistency_weight(iter_num // 150)
                loss_stu, outputs_weak = matchnet_train(args, weak_batch, strong_batch, label_batch, matchnet, ce_loss, dice_loss, consistency_weight)
                optimizer.zero_grad()
                loss_stu.backward()
                optimizer.step() 

                ###################################: logging and save matchnet model
                best_performance = write_log_match(args, epoch_num, iter_num, best_performance, base_lr, max_iterations, optimizer, writer, 
                                             consistency_weight, weak_batch, label_batch, outputs_weak, valloader,db_val, matchnet, sam, loss_stu)
            ## Medsam: finish one epoch
            best_loss = write_log_sam(epoch_num, epoch_loss, best_loss, sam, lr_scheduler_sam, optimizer_sam, snapshot_sam_path)
            ## MatchNet: finish one epoch

        else: # interative learning between medsam and matchnet
            epoch_loss = [1e10 for _ in range(len(trainloader))]
            # load training samples
            for step, sampled_batch in enumerate(trainloader):
                iter_num = iter_num + 1
                # load data
                weak_batch, strong_batch, label_batch, label_bbox = (
                    sampled_batch["image_weak"],
                    sampled_batch["image_strong"],
                    sampled_batch["label"], 
                    sampled_batch["bbox"],
                )
                weak_batch, strong_batch, label_batch, label_bbox = (
                    weak_batch.to(cuda_num),
                    strong_batch.to(cuda_num),
                    label_batch.to(cuda_num),
                    label_bbox.to(cuda_num)
                )
                ####################################: for matchnet
                consistency_weight = get_current_consistency_weight(iter_num // 150)
                outputs_strong, outputs_strong_soft,  outputs_weak, outputs_weak_soft, pseudo_outputs = matchnet_output(args, weak_batch, strong_batch, matchnet)

                ####################################: for MedSAM
                # finetune medsam with labeled data
                img_w_lb = weak_batch[: args.labeled_bs, ::] # Bx1x256x256
                img_w_lb_sam = torch.cat((img_w_lb, img_w_lb, img_w_lb), dim=1) # Bx3x256x256
                img_w_gt = label_batch[: args.labeled_bs, ::] # Bx256x256
                #  box from gt
                img_w_bbox = label_bbox[: args.labeled_bs, ::].unsqueeze(1)
                medsam_logit, iou_pred = sam(img_w_lb_sam, img_w_bbox)
                medsam_mask = torch.sigmoid(medsam_logit) > 0.5
                medsam_logit = medsam_logit.squeeze(1)
                
                # box from matchnet for unlabeled data
                img_w_ul = weak_batch[args.labeled_bs:, ::] # Bx1x256x256
                img_w_ul_sam = torch.cat((img_w_ul, img_w_ul, img_w_ul), dim=1) # Bx3x256x256
                # get box prompt: Bx1x4
                img_w_bbox_ul = get_bbox256_cv(pseudo_outputs, bbox_shift=args.bbox_shift)
                # presudo-labels
                medsam_logit_ul, iou_pred_ul = sam(img_w_ul_sam, img_w_bbox_ul)
                medsam_mask_ul = torch.sigmoid(medsam_logit_ul) > 0.5 # Bx1x256x256
                medsam_logit_ul = medsam_logit_ul.squeeze(1) # Bx256x256

                ###############################: calculate loss
                ## for matchnet
                # supervised loss
                sup_loss = ce_loss(outputs_weak[: args.labeled_bs], label_batch[: args.labeled_bs].long(),) + \
                                dice_loss(outputs_weak_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1))
                # complementary loss and adaptive sample weight for negative learning
                comp_loss, as_weight = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft, ce_loss=ce_loss)
                # unsupervised loss
                unsup_loss = (
                    ce_loss(outputs_strong[args.labeled_bs :], medsam_mask_ul.squeeze(1).long())
                    + dice_loss(outputs_strong_soft[args.labeled_bs :], medsam_mask_ul)
                    + as_weight * comp_loss
                )
                # total loss
                loss_stu = sup_loss * 2 + consistency_weight * unsup_loss

                ## for samus
                ## for unlabeled data
                loss_seg_w_sam_ul = dice_loss_sam(medsam_logit_ul, pseudo_outputs)
                loss_ce_w_sam_ul = ce_loss_sam(medsam_logit_ul, pseudo_outputs.float())
                loss_mask_w_sam_ul = loss_seg_w_sam_ul + loss_ce_w_sam_ul
                iou_gt_ul = cal_iou(medsam_mask_ul, pseudo_outputs.bool())
                loss_iou_w_sam_ul = iou_loss_sam(iou_pred_ul, iou_gt_ul)
                total_loss_w_sam_ul = loss_mask_w_sam_ul * 2 + loss_iou_w_sam_ul

                ## for labeled data
                loss_seg_w_sam = dice_loss_sam(medsam_logit, img_w_gt)
                loss_ce_w_sam = ce_loss_sam(medsam_logit, img_w_gt.float())
                loss_mask_w_sam = loss_seg_w_sam + loss_ce_w_sam
                iou_gt = cal_iou(medsam_mask, img_w_gt.bool())
                loss_iou_w_sam = iou_loss_sam(iou_pred, iou_gt)
                total_loss_w_sam = loss_mask_w_sam * 2 + loss_iou_w_sam

                loss_sam = total_loss_w_sam_ul*0.5 + loss_iou_w_sam
                epoch_loss[step] = loss_sam.item()
                # print("total_loss_w_sam_ul:{}, loss_iou_w_sam: {}".format(total_loss_w_sam_ul, loss_iou_w_sam))
                ## update student model
                optimizer.zero_grad()
                loss_stu.backward()
                optimizer.step()
                ## update sam model
                optimizer_sam.zero_grad()
                loss_sam.backward()
                optimizer_sam.step()
                
                ###################################: logging and save matchnet model
                best_performance = write_log_match(args, epoch_num, iter_num, best_performance, base_lr, max_iterations, optimizer, writer, 
                                             consistency_weight, weak_batch, label_batch, outputs_weak, valloader,db_val, matchnet, sam, loss_stu)

            ## Medsam: finish one epoch
            best_loss = write_log_sam(epoch_num, epoch_loss, best_loss, sam, lr_scheduler_sam, optimizer_sam, snapshot_sam_path)
            ## MatchNet: finish one epoch                



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

    snapshot_match_path = "./checkpoint/{}_{}_labeled_bs{}_shift_box{}/{}".format(
        args.exp, args.labeled_num, args.batch_size, args.bbox_shift ,args.model)
    if not os.path.exists(snapshot_match_path):
        os.makedirs(snapshot_match_path)

    snapshot_sam_path = "{}/{}".format(
        snapshot_match_path, args.sam_model)
    if not os.path.exists(snapshot_sam_path):
        os.makedirs(snapshot_sam_path)    

    logging.basicConfig(
        filename=snapshot_match_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_match_path, snapshot_sam_path)