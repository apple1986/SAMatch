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
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, util
from val_2D import test_single_volume
from segment_anything import sam_model_registry
from importlib import import_module
from model_sam.model_dict import get_model
from medpy import metric
from utils.generate_prompts import get_click_prompt,get_click_prompt_1
parser = argparse.ArgumentParser()

parser.add_argument('--adapt_ckpt', type=str, default='/home/gxu/proj1/smatch/SAMUS_3labeled.pth', help='The checkpoint after adaptation')
parser.add_argument('--ckpt', type=str, default='/home/gxu/proj1/smatch/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
parser.add_argument('--modelname', type=str, default='SAMUS')


parser.add_argument("--root_path", type=str, default="/home/gxu/proj1/smatch/ACDC", help="Name of Experiment")
parser.add_argument("--exp", type=str, default="ACDC/FixMatch_std_F2_case1", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4, help="output channel of network")
parser.add_argument("--load", default=False, action="store_true", help="restore previous checkpoint")
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.95,
    help="confidence threshold for using pseudo-labels",
)

parser.add_argument("--labeled_bs", type=int, default=2, help="labeled_batch_size per gpu")
# parser.add_argument('--labeled_num', type=int, default=136,
parser.add_argument("--labeled_num", type=int, default=1, help="labeled data")
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")
args = parser.parse_args()



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice#, hd95
    else:
        return 0
def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {
            "1":16,
            "3": 68,
            "7": 136,
            "14": 256,
            "21": 396,
            "28": 512,
            "35": 664,
            "140": 1312,
        }
    elif "Prostate":
        ref_dict = {
            "2": 27,
            "4": 53,
            "8": 120,
            "12": 179,
            "16": 256,
            "21": 312,
            "42": 623,
        }
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
                args.batch_size//2,
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
            comp_labels[args.batch_size//2:],
        )
        return comp_loss, as_weight

    def normalize(tensor):
        min_val = tensor.min(1, keepdim=True)[0]
        max_val = tensor.max(1, keepdim=True)[0]
        result = tensor - min_val
        result = result / max_val
        return result


    def findMax(predictions):
        max_values = []
        max_coords = []

        for i in range(predictions.shape[0]):
            # 找到每个 224x224 图像的最大值及其一维索引
            flat_image = predictions[i].view(-1)
             # 找到 1D 向量中的最大值及其索引
            max_value, max_index = torch.max(flat_image, dim=0)
            
            # 将一维索引转换为二维坐标
            max_row = max_index // predictions.size(2)  # 计算行号
            max_col = max_index % predictions.size(2)   # 计算列号
            
            # 存储结果
            #max_values.append(max_value.item())
            max_coords.append((max_row.item(), max_col.item()))
        return max_coords
    def find_random_one(mask,i):
        # 找到所有值为1的坐标
        pt = []
        lab = []
        for s in range(mask.shape[0]):
            coords = torch.nonzero(mask[s] == i, as_tuple=False)
            if coords.size(0) == 0:
                coord = torch.tensor([0,0]).cuda()
                lab.append(torch.tensor([0]).cuda())
            else:
            # 随机选择一个坐标
                random_idx = random.randint(0, coords.size(0) - 1)
                coord = coords[random_idx]
                lab.append(torch.tensor([i]).cuda())
            pt.append(coord)
        return pt,lab
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

    model = create_model()
    # create model for ema (this model produces pseudo-labels)
    ema_model = create_model(ema=True)
    
     ######SAM
    sam_net = get_model(args.modelname, args=args)
    sam_net.cuda()
    model.train()
    # checkpoint = torch.load(args.adapt_ckpt)
    # new_state_dict = {}
    # for k,v in checkpoint.items():
    #     if k[:7] == 'module.':
    #         new_state_dict[k[7:]] = v
    #     else:
    #         new_state_dict[k] = v
    # sam_net.load_state_dict(new_state_dict)
#########################################

    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer1 = optim.SGD(sam_net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # if restoring previous models:
    if args.load:
        try:
            # check if there is previous progress to be restored:
            logging.info(f"Snapshot path: {snapshot_path}")
            iter_num = []
            for filename in os.listdir(snapshot_path):
                if "model_iter" in filename:
                    basename, extension = os.path.splitext(filename)
                    iter_num.append(int(basename.split("_")[2]))
            iter_num = max(iter_num)
            for filename in os.listdir(snapshot_path):
                if "model_iter" in filename and str(iter_num) in filename:
                    model_checkpoint = filename
        except Exception as e:
            logging.warning(f"Error finding previous checkpoints: {e}")

        try:
            logging.info(f"Restoring model checkpoint: {model_checkpoint}")
            model, optimizer, start_epoch, performance = util.load_checkpoint(
                snapshot_path + "/" + model_checkpoint, model, optimizer
            )
            logging.info(f"Models restored from iteration {iter_num}")
        except Exception as e:
            logging.warning(f"Unable to restore model checkpoint: {e}, using new model")

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    # set to train
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    sam_loss = CrossEntropyLoss()
    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0

    iter_num = int(iter_num)

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    
    # for i_batch, sampled_batch in enumerate(valloader):
    #         weak_batch,  label_batch = (
    #             sampled_batch["image"],
    #             sampled_batch["label"],
    #         )
    #         weak_batch,  label_batch = (weak_batch.cuda().permute(1,0,2,3).to(torch.float32),
    #             label_batch.cuda().permute(1,0,2,3))
            
    #         dices = []
            
    #         for i in range(1,4):
    #             max_index = find_random_one(label_batch.squeeze(1),i)
    #             pt = get_click_prompt_1(max_index)
    #             pred = sam_net(weak_batch,pt,None)
    #             predict_masks = pred['masks']
    #             predict_masks = torch.sigmoid(predict_masks)
    #             predict = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
    #             #segs = predict[:, 0, :, :] > 0.5  # (b, h, w)
    #             predict[predict > 0.5] = 1
    #             predict[predict != 1] = 0
    #             # pred_mask[pred_mask!=1] = 0
    #             label_batch[label_batch==i] = 1
    #             label_batch[label_batch!=i] = 0
    #             label = label_batch[:,0,:,:].detach().cpu().numpy()
    #             dice = calculate_metric_percase(predict[:,0,:,:][0],label[0])
    #             dices.append(dice)
    #         print(dice)


    for epoch_num in iterator:

        for i_batch, sampled_batch in enumerate(trainloader):
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )

            # outputs for model
            outputs_weak = model(weak_batch)
            outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
            outputs_strong = model(strong_batch[args.labeled_bs:])
            outputs_strong_soft = torch.softmax(outputs_strong, dim=1)
            pseudo_mask = (normalize(outputs_weak_soft) > args.conf_thresh).float()
            outputs_weak_masked = outputs_weak_soft * pseudo_mask
            pseudo_outputs = torch.argmax(outputs_weak_masked[args.labeled_bs :].detach(), dim=1, keepdim=False)
            # minmax normalization for softmax outputs before applying mask
            #sam_weak = outputs_weak_soft[12:] 
            #with torch.no_grad():
                #n = random.randint(12,24)
            pred = sam_net(weak_batch,None)
            predict = pred['masks']
            sam_pred = torch.softmax(predict, dim=1)
            out = torch.argmax(sam_pred[args.batch_size//2:], dim=1).squeeze(0).squeeze(0)
            # metric_1 = (calculate_metric_percase(
            #         (out.cpu().numpy())==1, (label_batch[12:].cpu().numpy()) == 1))
            # metric_2 = calculate_metric_percase(
            #         (out.cpu().numpy())==2, (label_batch[12:].cpu().numpy()) == 2)
            # metric_3 = calculate_metric_percase(
            #         (out.cpu().numpy())==3, (label_batch[12:].cpu().numpy()) == 3)
            # print(metric_1,metric_2,metric_3)
                # for i in range(1,4):
                #     pt_mask = sam_weak[:,i:i+1,:,:]
                #     max_index = findMax(pt_mask.squeeze(1))
                #     pt = get_click_prompt_1(max_index,[i]*12)
                #     pred = sam_net(weak_batch[12:],pt,None)
                #     predict_masks = torch.sigmoid(pred['masks'])
                    
                #     predict_masks[predict_masks> 0.5]  = 1*i # (b, h, w)
                #     predict_masks[predict_masks< 0.5]  = 0 
                    # for seg in segs:
                    #     pro = calculate_metric_percase(
                    #      (seg)==1, (label_batch[12:].cpu().numpy()) == i)
                    #     print("第",i,"类",pro)             
            # out = torch.argmax(torch.softmax(
            #     predict_masks, dim=1), dim=1).squeeze(0).squeeze(0)
                
                # metric = 0
                # metric = calculate_metric_percase(
                #     out.cpu().numpy() == 1, label_batch[12:].cpu().numpy() == s)
                # print(metric)
               
               

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # pseudo_mask = torch.argmax(outputs_weak_soft[args.labeled_bs :].detach(), dim=1, keepdim=False)
            # sorted_n, indices = torch.sort(outputs_weak_soft, dim=1)

            # supervised loss
            sup_loss = ce_loss(outputs_weak[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(
                outputs_weak_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )

            # complementary loss and adaptive sample weight for negative learning
            comp_loss, as_weight = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft)

            # # unsupervised loss
            # unsup_loss = (
            #     ce_loss(outputs_strong, pseudo_outputs)
            #     + dice_loss(outputs_strong_soft, pseudo_outputs.unsqueeze(1))
            #     + as_weight * comp_loss
            # )
            unsup_loss_1 = (
                ce_loss(outputs_strong, out)
                + dice_loss(outputs_strong_soft, out.unsqueeze(1))
                + as_weight * comp_loss
            )
            
            loss_sam = (dice_loss(sam_pred[args.labeled_bs:],pseudo_outputs.unsqueeze(1))*consistency_weight + sam_loss(predict[:args.labeled_bs],label_batch[:][: args.labeled_bs].long())
                                                    +dice_loss(sam_pred[: args.labeled_bs],label_batch[: args.labeled_bs].unsqueeze(1))
                                                    )
            
            loss = sup_loss + consistency_weight * (unsup_loss_1)#+unsup_loss_1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            optimizer1.zero_grad()
            loss_sam.backward()
            optimizer1.step()
            # update ema model
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

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
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model,
                        classes=num_classes,
                    )
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
                sam_pth = os.path.join(snapshot_path,  "SAM" + str(iter_num) + ".pth")
                util.save_checkpoint(epoch_num, model, optimizer, loss, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                torch.save(sam_net.state_dict(),sam_pth, _use_new_zipfile_serialization=False)
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

    snapshot_path = "/home/gxu/proj1/smatch/checkpoint/{}_{}/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + "/code"):
    #     shutil.rmtree(snapshot_path + "/code")
    # shutil.copytree('.',snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"]))

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
