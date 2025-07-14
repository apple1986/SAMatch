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

################# for sam
from model_sam.model_dict import get_model
from utils.loss_functions.sam_loss import Mask_DC_and_BCE_loss_ap
import monai
import cv2

#################

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/MRliver", help="Name of Experiment")
parser.add_argument("--save_path", type=str, default="./checkpoint", help="Name of Experiment")
parser.add_argument("--exp", type=str, default="MRliver/MedSam_ft", help="experiment_name: finetune medsam")
parser.add_argument("--sam_model", type=str, default="medsam", help="model_name") 
## para setting
parser.add_argument('--cuda_num', type=str,  default="cuda:0", help='random seed')
parser.add_argument("--labeled_num", type=int, default=5, 
                    help="how many labeled cases are used. Note that each case includes some slices")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
# parser.add_argument("-bbox_shift", type=int, default=5, help="Perturbation to bounding box coordinates during training.")
parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
parser.add_argument("--labeled_bs", type=int, default=2, help="labeled_batch_size per gpu, how may labeded slice used for training")
parser.add_argument("--base_lr_sam", type=float, default=5e-5, help="segmentation network learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=2, help="output channel of network")

# for pretrained medsam
parser.add_argument('--ckpt_samus', type=str, 
                    default='/home/gxu/proj1/smatch/checkpoint/SAM/lite_medsam.pth', 
                    help='Fine-tuned SAM')
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

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

###################build model
class MedSAM_Lite(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks


## load medsam
def load_medsam(args):
    medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
    )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )

    medsam_lite_model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )
    # load pretrained medsam
    lite_medsam_checkpoint = torch.load(args.ckpt_samus, map_location='cpu', weights_only=True)
    medsam_lite_model.load_state_dict(lite_medsam_checkpoint, strict=True)
    medsam_lite_model.to(args.cuda_num)
    print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
    return medsam_lite_model

##################### for medsam: get prompt box
def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image
    Shape: 255 x 256
    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256

def get_bbox256_torch(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image
    Shape: 255 x 256
    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
        bounding box coordinates in the resized image
    """
    B, H, W = mask_256.shape
    bboxes256 = torch.ones((B, 1, 4)).to(mask_256.device) * (-100)
    for n in range(B):
        pd_one = mask_256[n,:,:]
        idx_fg = torch.argwhere(pd_one > 0.5)
        if (idx_fg.sum() > 0):
            x_min, x_max = torch.min(idx_fg[:,1]), torch.max(idx_fg[:, 1])
            y_min, y_max = torch.min(idx_fg[:,0]), torch.max(idx_fg[:, 0])
            x_min = max(0, x_min - bbox_shift)
            x_max = min(W, x_max + bbox_shift)
            y_min = max(0, y_min - bbox_shift)
            y_max = min(H, y_max + bbox_shift)
            bboxes256[n, 0, :] = torch.tensor([x_min, y_min, x_max, y_max])

    return bboxes256

def get_bbox256_cv(mask_256, bbox_shift=3):
    B, H, W = mask_256.shape
    binary_mask = mask_256.detach().cpu().numpy()
    bboxes256 = np.ones((B, 1, 4))* (-100)#.to(mask_256.device) * (-100)
    for n in range(B):
        pd_one = binary_mask[n, :, :].astype(np.uint8)
        if (pd_one.sum()> 0):
            # Find contours in the binary mask
            contours, _ = cv2.findContours(pd_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize variables to keep track of the largest bounding box
            max_area = 0           
            for contour in contours:
                # Get the bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # Update the largest bounding box if this one is bigger
                if area > max_area:
                    max_area = area
                    x_min = max(0, x - bbox_shift)
                    x_max = min(W, x + w + bbox_shift)
                    y_min = max(0, y - bbox_shift)
                    y_max = min(H, y + h + bbox_shift)
                    bboxes256[n, 0, :] = np.array([x_min, y_min, x_max, y_max])
    bboxes256 = torch.tensor(bboxes256).to(mask_256.device)

    return bboxes256    

#########################SAM setting
def sam_set(medsam_lite_model, lr=5e-5):
    optimizer = optim.AdamW(
                medsam_lite_model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=5,
        cooldown=0
    )
    dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    iou_loss = nn.MSELoss(reduction='mean')
    return optimizer, lr_scheduler, dice_loss, ce_loss, iou_loss

def cal_iou(result, reference):
    
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    
    iou = intersection.float() / (union.float()+1e-9)
    
    return iou.unsqueeze(1)

###################################
def train(args, snapshot_match_path, snapshot_sam_path):
    ## 1. hyper-para setting
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
        base_dir=args.data_path,
        split="train",
        num=None,
        transform=transforms.Compose([WeakStrongAugment(args.patch_size)]),
    )
    db_val = BaseDataSets(base_dir=args.data_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.data_path, args.labeled_num)
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
    sam = load_medsam(args)
    # load optim,loss, lr
    optimizer_sam, lr_scheduler_sam, dice_loss_sam, ce_loss_sam, iou_loss_sam = sam_set(sam, lr=lr_sam)

    # set to train
    sam.train()
    # write record
    writer = SummaryWriter(snapshot_match_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    # cal training iteration numbers
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    print("\nstart_epoch:{},  max_epoch: {}".format(start_epoch, max_epoch))
    best_loss = 1e10
    for epoch_num in iterator:
        print(epoch_num)
        epoch_loss = [1e10 for _ in range(len(trainloader))]
        # load training samples
        for step, sampled_batch in enumerate(trainloader):
            # load data
            weak_batch, strong_batch, label_batch, label_bbox, case_id = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label"], 
                sampled_batch["bbox"],
                sampled_batch["case"],
            )
            weak_batch, strong_batch, label_batch, label_bbox = (
                weak_batch.to(cuda_num),
                strong_batch.to(cuda_num),
                label_batch.to(cuda_num),
                label_bbox.to(cuda_num)
            )
            # print(sampled_batch["case"])
            ####################################: for medsam
            # finetune medsam with labeled data
            img_w_lb = weak_batch[: args.labeled_bs, ::] # Bx1x256x256
            img_w_lb_sam = torch.cat((img_w_lb, img_w_lb, img_w_lb), dim=1) # Bx3x256x256
            img_w_gt = label_batch[: args.labeled_bs, ::] # Bx256x256
            #  box from gt
            img_w_bbox = label_bbox[: args.labeled_bs, ::].unsqueeze(1)
            medsam_logit, iou_pred = sam(img_w_lb_sam, img_w_bbox)
            medsam_mask = torch.sigmoid(medsam_logit) > 0.5 # torch.Size([4, 1, 256, 256])
            medsam_logit = medsam_logit.squeeze(1) # torch.Size([4, 256, 256])
            # print(f'{case}_{idx}, box: {box256}, predicted iou: {np.round(iou_pred.item(), 4)}')

            # cal loss
            loss_seg_w_sam = dice_loss_sam(medsam_logit, img_w_gt)
            loss_ce_w_sam = ce_loss_sam(medsam_logit, img_w_gt.float())
            loss_mask_w_sam = loss_seg_w_sam + loss_ce_w_sam
            iou_gt = cal_iou(medsam_mask, img_w_gt.bool())
            loss_iou_w_sam = iou_loss_sam(iou_pred, iou_gt)
            total_loss_w_sam = loss_mask_w_sam * 2 + loss_iou_w_sam
            # print('loss_mask_w_sam: {}, loss_iou_w_sam: {}'.format(loss_mask_w_sam, loss_iou_w_sam))
            epoch_loss[step] = total_loss_w_sam.item()
            # update gradient
            optimizer_sam.zero_grad()
            total_loss_w_sam.backward()
            optimizer_sam.step()
            
            # update iteration number and save training log
            iter_num = iter_num + 1
            writer.add_scalar("lr", lr_sam, iter_num)
            writer.add_scalar("loss/model_loss", total_loss_w_sam, iter_num)
            logging.info("iteration %d : model loss : %f" % (iter_num, total_loss_w_sam.item()))

            if iter_num % 50 == 0:
                image = weak_batch[1, 0:1, :, :]
                writer.add_image("train/Image", image, iter_num)
                writer.add_image("train/model_Prediction", medsam_mask[1, 0:1, :, :] * 50, iter_num)

                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)
                
        ## Medsam: finish one epoch
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


    snapshot_sam_path = "./checkpoint/{}_{}_labeled_bs{}/{}".format(
        args.exp, args.labeled_num, args.batch_size, args.sam_model)
    if not os.path.exists(snapshot_sam_path):
        os.makedirs(snapshot_sam_path)    

    logging.basicConfig(
        filename=snapshot_sam_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_sam_path, snapshot_sam_path)