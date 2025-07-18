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
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset_busi import (
    BaseDataSets,
    RandomGenerator,
    TwoStreamBatchSampler,
    WeakStrongAugment,
)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, util
from val_2D import test_single_volume_BUSI
## medsam
from networks.medsam import MedSAM_Lite, load_medsam
from utils.get_prompts import get_bbox256_cv, get_bbox256_torch


LABELED_NUM = "30"
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="./data/BUSI", help="path of dataset")
parser.add_argument("--exp", type=str, default="BUSI/FixMatch_MedSAM_both_bbox30", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--labeled_num", type=str, default=LABELED_NUM, help="labeled number of cases")
parser.add_argument("--bbox_shift", type=int, default=30, help="bbox shift")
parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
parser.add_argument("--labeled_bs", type=int, default=2, help="labeled_batch_size per gpu")
parser.add_argument("--conf_thresh", type=float, default=0.95, help="confidence threshold for using pseudo-labels",)
parser.add_argument('--cuda_num', type=str,  default="cuda:0", help='cuda index')
parser.add_argument('--ckpt_sam', type=str, 
                    default='/data/maia/gpxu/proj1/samatch/checkpoint/BUSI/MedSam_ft_'+str(LABELED_NUM)+'_labeled_bs4/medsam/medsam_lite_best.pth', 
                    help='Fine-tuned SAM')
parser.add_argument('--ckpt_fix', type=str, 
                    default='/data/maia/gpxu/proj1/samatch/checkpoint/BUSI/FixMatch_standard_augs_'+str(LABELED_NUM)+'_labeled_bs4/unet/unet_best_model.pth', 
                    help='Fine-tuned SAM')

parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--base_lr_sam", type=float, default=5e-4, help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=2, help="output channel of network")
parser.add_argument("--load", default=False, action="store_true", help="restore previous checkpoint")

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

    elif "BUSI" in dataset:
        ref_dict = {"10": 10, "30": 30, "50": 50, "70": 70, "90": 90, "377": 377}

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

def load_fixmatch(ckpt_match, model):
    # load_weight = torch.load(ckpt_match, weights_only=False, map_location=args.cuda_num)["model"]
    load_weight = torch.load(ckpt_match, weights_only=True , map_location=args.cuda_num)["state_dict"]
    model.load_state_dict(load_weight)
    return model

def train(args, snapshot_path):
    base_lr = args.base_lr # learning rate
    num_classes = args.num_classes # class of objects
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, cuda_num=args.cuda_num)
        if ema: # teacher model
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def get_comp_loss(weak, strong):
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

    # student model
    model = create_model()
    # load model
    model = load_fixmatch(args.ckpt_fix, model)

    # create model for ema (this model produces pseudo-labels): teacher model
    # ema_model = create_model(ema=True)

    # load the pretrained sam model
    sam = load_medsam(args.ckpt_sam, args.cuda_num)
    sam.to(args.cuda_num)
    sam.train()
    # load optim,loss, lr
    # optimizer_sam, lr_scheduler_sam = sam_set_uni(sam, lr=args.base_lr_sam)

    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_sam = optim.SGD(sam.parameters(), lr=args.base_lr_sam, momentum=0.9, weight_decay=0.0001) #base_lr_sam: 5e-5

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # set to train
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    dice_loss_sam = losses.DiceLoss_U(n_classes=2)

    writer = SummaryWriter(snapshot_path + "/log") # record the information to tensorboard
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0

    iter_num = int(iter_num)

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    img_temp_bbox_all = torch.zeros((args.labeled_bs, num_classes-1, 4)).to(args.cuda_num)
    for epoch_num in iterator:

        for i_batch, sampled_batch in enumerate(trainloader):
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.to(args.cuda_num),
                strong_batch.to(args.cuda_num),
                label_batch.to(args.cuda_num),
            )

            # outputs for model
            outputs_weak = model(weak_batch)
            outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
            outputs_strong = model(strong_batch)
            outputs_strong_soft = torch.softmax(outputs_strong, dim=1)

            # minmax normalization for softmax outputs before applying mask
            pseudo_mask = (normalize(outputs_weak_soft) > args.conf_thresh).float()
            outputs_weak_masked = outputs_weak_soft * pseudo_mask # get the thresholded prob map
            # get pseudo-labels for unlabeled data
            pseudo_outputs = torch.argmax(outputs_weak_masked[args.labeled_bs :].detach(), dim=1, keepdim=False)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # sam to generate psedudo-label as involves in training
            img_x = weak_batch[: args.labeled_bs]
            mask_x = label_batch[: args.labeled_bs]
            mask_x = mask_x.long()
            img_u_w = weak_batch[args.labeled_bs:] 
            img_x_3ch = torch.cat((img_x, img_x, img_x), dim=1) # 4x1xHxW
            img_u_w_3ch = torch.cat((img_u_w, img_u_w, img_u_w), dim=1)
            # img_sam = torch.cat((img_x_3ch, img_u_w_3ch),dim=0) # normalized # B3HW
            # ## there are three classes, so generate three boxes. THen select an box randomly for a prmpmt
            # for n in range(0, num_classes-1): # cls = 1, 2, 3
            #     img_temp_bbox_all[:, n, :] = get_bbox256_torch(mask_x==(n+1))[:,0,:] # Bx1x4
            # # get one box from gt mask           
            # idx = random.choice([0,1,2])
            # img_gt_bbox = img_temp_bbox_all[:, idx, :].unsqueeze(1) # Bx1x4
            # mask_x_one = (mask_x == (idx+1)).long() # notice the object class: 1, 2, 3      

            img_gt_bbox = get_bbox256_torch(mask_x==1)# Bx1x4

            # do predition, and generate a prob map of one class
            ## prediction 
            medsam_logit_x, _ = sam(img_x_3ch, img_gt_bbox)
            # medsam_mask_x = torch.sigmoid(medsam_logit_x) > 0.5 # 8x1xHxW
            pred_sam_pos = torch.sigmoid(medsam_logit_x) # B*1*H*W
            pred_sam_neg = 1 - pred_sam_pos
            pred_x_sam = torch.cat((pred_sam_neg, pred_sam_pos), dim=1)

            ## do three times for unlabeled data
            ## prediction 
            # with torch.no_grad():
            #     sam.eval()
            img_pd_bbox = get_bbox256_cv(pseudo_outputs==1, bbox_shift=args.bbox_shift)
            medsam_logit_1, _ = sam(img_u_w_3ch, img_pd_bbox)
            # img_pd_bbox = get_bbox256_cv(pseudo_outputs==2, bbox_shift=args.bbox_shift)
            # medsam_logit_2, _ = sam(img_u_w_3ch, img_pd_bbox)
            # img_pd_bbox = get_bbox256_cv(pseudo_outputs==3, bbox_shift=args.bbox_shift)
            # medsam_logit_3, _ = sam(img_u_w_3ch, img_pd_bbox)
            medsam_logit_1 = torch.sigmoid(medsam_logit_1)
            medsam_logit_0 = 1 - medsam_logit_1
            a_tmp = torch.cat((medsam_logit_0, medsam_logit_1), dim=1)
            
            # medsam_logit_all[:, 0, :, :] = (1 - torch.mean(medsam_logit_all[:, 1:, :, :], dim=1, keepdim=True)).squeeze(1) # for the background
            pred_w_sam = a_tmp.softmax(dim=1)
            mask_u_w_sam = pred_w_sam.argmax(dim=1)
            # labeled loss from sam
            # pred_x_sam: Bx2xHxW, mask_x: BxHxW
            loss_x_sam = (ce_loss(pred_x_sam, mask_x) + 
                        dice_loss_sam(pred_x_sam, mask_x.unsqueeze(1).float())) / 2.0
            # unlabed loss from sam
            loss_u_w_sam = (ce_loss(pred_w_sam, pseudo_outputs) + 
                        dice_loss(pred_w_sam, pseudo_outputs.unsqueeze(1).float())) / 2.0
            loss_sam = loss_x_sam + loss_u_w_sam * 0.25
            # epoch_loss[i] = loss_sam.item()
            # ## update sam model
            optimizer_sam.zero_grad()
            loss_sam.backward()
            optimizer_sam.step()



            # supervised loss: CE + Dice
            sup_loss = ce_loss(outputs_weak[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(
                outputs_weak_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )

            # complementary loss and adaptive sample weight for negative learning
            comp_loss, as_weight = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft)

            # unsupervised loss: CE+Dice+comp_loss
            unsup_loss = (
                ce_loss(outputs_strong[args.labeled_bs :], mask_u_w_sam)
                + dice_loss(outputs_strong_soft[args.labeled_bs :], mask_u_w_sam.unsqueeze(1))
                + as_weight * comp_loss
            )

            loss = sup_loss + consistency_weight * unsup_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update ema model
            # update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1

            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("consistency_weight/consistency_weight", consistency_weight, iter_num)
            writer.add_scalar("loss/model_loss", loss, iter_num)
            logging.info("iteration %d : model loss : %f" % (iter_num, loss.item()))
            if iter_num % 50 == 0:
                image = weak_batch[1, 0:1, :, :]
                writer.add_image("train/Image", image, iter_num)
                outputs_weak = torch.argmax(torch.softmax(outputs_weak, dim=1), dim=1, keepdim=True)
                writer.add_image("train/model_Prediction", outputs_weak[1, ...] * 50, iter_num)

                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    # metric_i = test_single_volume_BUSI(
                    #     sampled_batch["image"],
                    #     sampled_batch["label"],
                    #     model,
                    #     classes=num_classes,
                    #     cuda_num=args.cuda_num,
                    # )
                    metric_i = test_single_volume_BUSI(sampled_batch["image"], sampled_batch["label"], model, classes=2, 
                                                       patch_size=[256, 256], cuda_num=args.cuda_num,)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        "info/model_val_{}_dice".format(class_i + 1),
                        metric_list[class_i, 0],
                        iter_num,
                    )
                    writer.add_scalar(
                        "info/model_val_{}_hd95".format(class_i + 1),
                        metric_list[class_i, 1],
                        iter_num,
                    )

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar("info/model_val_mean_dice", performance, iter_num)
                writer.add_scalar("info/model_val_mean_hd95", mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "model_iter_{}_dice_{}.pth".format(iter_num, round(best_performance, 4)),
                    )
                    save_best = os.path.join(snapshot_path, "{}_best_model.pth".format(args.model))
                    util.save_checkpoint(epoch_num, model, optimizer, loss, save_mode_path)
                    util.save_checkpoint(epoch_num, model, optimizer, loss, save_best)

                logging.info(
                    "iteration %d : model_mean_dice : %f model_mean_hd95 : %f" % (iter_num, performance, mean_hd95)
                )
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, "model_iter_" + str(iter_num) + ".pth")
                util.save_checkpoint(epoch_num, model, optimizer, loss, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


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