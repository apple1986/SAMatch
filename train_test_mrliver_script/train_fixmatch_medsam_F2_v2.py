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
parser.add_argument("--root_path", type=str, default="./data/MRliver", help="Name of Experiment")
parser.add_argument("--save_path", type=str, default="./checkpoint", help="Name of Experiment")
parser.add_argument("--exp", type=str, default="MRliver/FixMatch_medsam", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="model_name") 
parser.add_argument("--sam_model", type=str, default="medsam", help="model_name") 
parser.add_argument("--iter_num", type=int, default=3000, help="when train sam after matchnet")
## select gpu: setting the net_factory and test_single_volume function
parser.add_argument('--cuda_num', type=str,  default="cuda:1", help='random seed')
parser.add_argument( "--conf_thresh", type=float,default=0.5, help="confidence threshold for using pseudo-labels")
parser.add_argument("--max_iterations", type=int, default=60000, help="maximum epoch number to train")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--base_lr_sam", type=float, default=5e-4, help="segmentation network learning rate")
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

######################## load pretrained medsam-lit
def load_samus(args):
    model = get_model("SAMUS", args=args, opt=None)
    model.to(args.cuda_num)
    # model.eval()
    checkpoint = torch.load(args.ckpt_samus, weights_only=True)
    #------when the load model is saved under multiple GPU
    new_state_dict = {}
    for k,v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    # model.eval()
    return model

###################build model
##########build model
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


## load sam
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

    lite_medsam_checkpoint = torch.load(args.ckpt_samus, map_location='cpu', weights_only=True)
    medsam_lite_model.load_state_dict(lite_medsam_checkpoint, strict=True)
    medsam_lite_model.to(args.cuda_num)
    # %%
    print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
    # %%
    return medsam_lite_model

################################### find the high confident point
def find_position(unlabel_pd):
    B, H, W = unlabel_pd.shape 
    temp_pd = unlabel_pd.view(B, -1)           
    M = temp_pd.argmax(1) # B
    ## if the gt is blank, it will result in error prompt
    is_null = temp_pd.sum(dim=1) < args.conf_thresh
    if is_null.sum() > 0:
        M[is_null] = 0
        # M[is_null] = torch.tensor(32896, device=M.device)
    #     print("Here")
    # else:
    #     print("is not null")

    idx = torch.cat(((M / H).view(-1, 1), (M % W).view(-1, 1)), dim=1).long()
    idx[:, [0,1]] = idx[:, [1,0]] # (Y,X) --> (X,Y)
    input_points = idx.unsqueeze(1)
    # input_labels = torch.ones((B, 1), device=args.cuda_num)
    input_labels = (~is_null).float().reshape(B, -1) #torch.ones((B, 1), device=args.cuda_num)
    point_label = (input_points, input_labels)
    return point_label

def random_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], point_label

def generate_click_prompt(img, msk, pt_label = 1):
    # return: img, prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0) # b 2
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)  # c b 2
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1) # b 2 d
    msk = torch.stack(msk_list, dim=-1) # b h w d
    msk = msk.unsqueeze(1) # b c h w d
    return img, pt, msk #[b, 2, d], [b, c, h, w, d]

def generate_unique_random_numbers(n, start, end):
    # generate n number in [start, end]
    return random.sample(range(start, end + 1), n)

def find_point_label(gt_sam):
    # get 10 points for gt
    B, _, _ = gt_sam.shape
    points_coord = torch.zeros((B, 10, 2), device=gt_sam.device)
    points_label = torch.zeros((B, 10), device=gt_sam.device)
    SEL_PT_NUM = 1 # how many positive poinsts are selected?
    for n in range(B):
        gt_one = gt_sam[n,:,:] # HW
        idx_fg = torch.argwhere(gt_one == 1) # 1 is the class label
        idx_bg = torch.argwhere(gt_one != 1)
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
            random_numbers = generate_unique_random_numbers(SEL_PT_NUM, 0, len(idx_fg)-1)
            # foreground: points and labels
            points_coord[n,:SEL_PT_NUM,:] = idx_fg[random_numbers]
            points_label[n,:SEL_PT_NUM] = 1
            # backgrouond: points and labels
            random_numbers = generate_unique_random_numbers(10-SEL_PT_NUM, 0, len(idx_bg)-1)
            points_coord[n,SEL_PT_NUM:,] = idx_bg[random_numbers]
            points_label[n,SEL_PT_NUM:] = 0
    return points_coord, points_label

def find_point_label_pseudo(pd_sam):
    # input: pd_sam is a probability map
    # get 10 points for pseudo-label: 1 for highest probability, 9 for bg
    B, _, _ = pd_sam.shape
    points_coord = torch.zeros((B, 10, 2), device=pd_sam.device)
    points_label = torch.zeros((B, 10), device=pd_sam.device)
    for n in range(B):
        pd_one = pd_sam[n,:,:] # HW
        idx_fg = torch.argwhere(pd_one > 0.5).to(device=pd_sam.device) # 1 is the class label, 0.5 is threshold
        idx_bg = torch.argwhere(pd_one < 0.5).to(device=pd_sam.device) 
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
##################### for medsam
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
            # print(idx_fg)
            # idx_bg = torch.argwhere(pd_one < 0.5)
            x_min, x_max = torch.min(idx_fg[:,1]), torch.max(idx_fg[:, 1])
            y_min, y_max = torch.min(idx_fg[:,0]), torch.max(idx_fg[:, 0])
            x_min = max(0, x_min - bbox_shift)
            x_max = min(W, x_max + bbox_shift)
            y_min = max(0, y_min - bbox_shift)
            y_max = min(H, y_max + bbox_shift)
            bboxes256[n, 0, :] = torch.tensor([x_min, y_min, x_max, y_max])
            # print(bboxes256)
            # print("*"*50)
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


######################train setting
# cread match
def create_model(ema=False):
    model = net_factory(net_type=args.model, in_chns=1, 
                        class_num=args.num_classes, cuda_num=args.cuda_num)
    if ema:
        for param in model.parameters():
            param.detach_()
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
    sam = load_medsam(args)
    # load optim,loss, lr
    optimizer_sam, lr_scheduler_sam, dice_loss_sam, ce_loss_sam, iou_loss_sam = sam_set(sam, lr=lr_sam)

    # load match model
    matchnet = create_model()
    # create model for ema (this model produces pseudo-labels)
    ema_model = create_model(ema=True)
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
                
                ####################################: for medsam
                # finetune medsam with labeled data
                img_w_lb = weak_batch[: args.labeled_bs, ::] # Bx1x256x256
                img_w_lb_sam = torch.cat((img_w_lb, img_w_lb, img_w_lb), dim=1) # Bx3x256x256
                img_w_gt = label_batch[: args.labeled_bs, ::] # Bx256x256
                #  box from gt
                img_w_bbox = label_bbox[: args.labeled_bs, ::].unsqueeze(1)
                medsam_logit, iou_pred = sam(img_w_lb_sam, img_w_bbox)
                medsam_mask = torch.sigmoid(medsam_logit) > 0.5
                medsam_logit = medsam_logit.squeeze(1)
                # print(f'{case}_{idx}, box: {box256}, predicted iou: {np.round(iou_pred.item(), 4)}')
                # medsam_mask_c = medsam_mask.detach().cpu().numpy()
                # img_w_gt_c = img_w_gt.detach().cpu().numpy()
                # import matplotlib
                # matplotlib.use('TkAgg')
                # import matplotlib.pyplot as plt
                # NUM = 1
                # plt.imshow(medsam_mask_c[NUM,0,:,:]*255, cmap="gray"); plt.show()
                # plt.figure(); 
                # plt.imshow(img_w_gt_c[NUM,:,:]*255, cmap="gray"); plt.show()

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
                ####################################: for matchnet
                # outputs for model
                outputs_weak = matchnet(weak_batch) # without normalize
                outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
                outputs_strong = matchnet(strong_batch)
                outputs_strong_soft = torch.softmax(outputs_strong, dim=1)

                # minmax normalization for softmax outputs before applying mask
                pseudo_mask = (normalize(outputs_weak_soft) > args.conf_thresh).float()
                outputs_weak_masked = outputs_weak_soft * pseudo_mask
                pseudo_outputs = torch.argmax(outputs_weak_masked[args.labeled_bs :].detach(), dim=1, keepdim=False)
                consistency_weight = get_current_consistency_weight(iter_num // 150)
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
                ## update student model
                loss_stu = sup_loss + consistency_weight * unsup_loss
                optimizer.zero_grad()
                loss_stu.backward()
                optimizer.step() 
                # update learning rate
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_  
                # update iteration number and save training log
                iter_num = iter_num + 1
                writer.add_scalar("lr", lr_, iter_num)
                writer.add_scalar("consistency_weight/consistency_weight", consistency_weight, iter_num)
                writer.add_scalar("loss/model_loss", loss_stu, iter_num)
                logging.info("iteration %d : model loss : %f" % (iter_num, loss_stu.item()))

                # update iteration number and save training log
                iter_num = iter_num + 1
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
                            classes=num_classes,
                            cuda_num=cuda_num,
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
                            snapshot_match_path,
                            "model_iter_{}_dice_{}.pth".format(iter_num, round(best_performance, 4)),
                        )
                        save_best = os.path.join(snapshot_match_path, "{}_best_model.pth".format(args.model))
                        util.save_checkpoint(epoch_num, matchnet, optimizer, loss_stu, save_mode_path)
                        util.save_checkpoint(epoch_num, matchnet, optimizer, loss_stu, save_best)
                        ## save sam
                        save_best_sam = os.path.join(snapshot_match_path, "best_sam_model.pth")
                        torch.save(sam.state_dict(), save_best_sam, _use_new_zipfile_serialization=False)

                    logging.info(
                        "iteration %d : model_mean_dice : %f model_mean_hd95 : %f" % (iter_num, performance, mean_hd95)
                    )
                    matchnet.train()

                if iter_num % 3000 == 0:
                    save_mode_path = os.path.join(snapshot_match_path, "model_iter_" + str(iter_num) + ".pth")
                    util.save_checkpoint(epoch_num, matchnet, optimizer, loss_stu, save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                
            ## Medsam: finish one epoch
            epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)            
            lr_scheduler_sam.step(epoch_loss_reduced)
            model_weights = sam.state_dict()
            checkpoint = {
                "model": model_weights,
                "epoch": epoch_num,
                "optimizer": optimizer.state_dict(),
                "loss": epoch_loss_reduced,
            }
            torch.save(checkpoint, os.path.join(snapshot_sam_path, "medsam_lite_latest.pth"))
            if epoch_loss_reduced < best_loss:
                print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
                best_loss = epoch_loss_reduced
                checkpoint["best_loss"] = best_loss
                torch.save(checkpoint, os.path.join(snapshot_sam_path, "medsam_lite_best.pth"))

            ## MatchNet: finish one epoch

        else: # interative learning between medsam and matchnet
            epoch_loss = [1e10 for _ in range(len(trainloader))]
            # load training samples
            for step, sampled_batch in enumerate(trainloader):
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
                # outputs for model
                outputs_weak = matchnet(weak_batch) # without normalize
                outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
                outputs_strong = matchnet(strong_batch)
                outputs_strong_soft = torch.softmax(outputs_strong, dim=1)

                # minmax normalization for softmax outputs before applying mask
                pseudo_mask = (normalize(outputs_weak_soft) > args.conf_thresh).float()
                # pseudo_mask = (normalize(outputs_weak_soft) > 0.5).float()
                outputs_weak_masked = outputs_weak_soft * pseudo_mask
                pseudo_outputs = torch.argmax(outputs_weak_masked[args.labeled_bs :].detach(), dim=1, keepdim=False)
                consistency_weight = get_current_consistency_weight(iter_num // 150)

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
                
                # box from matchnet
                img_w_ul = weak_batch[args.labeled_bs:, ::] # Bx1x256x256
                img_w_ul_sam = torch.cat((img_w_ul, img_w_ul, img_w_ul), dim=1) # Bx3x256x256
                # img_w_bbox_ul = get_bbox256_torch(pseudo_outputs) # Bx1x4
                img_w_bbox_ul = get_bbox256_cv(pseudo_outputs, args.bbox_shift)
                
                medsam_logit_ul, iou_pred_ul = sam(img_w_ul_sam, img_w_bbox_ul)
                medsam_mask_ul = torch.sigmoid(medsam_logit_ul) > 0.5
                medsam_logit_ul = medsam_logit_ul.squeeze(1)

                ## for match-net
                # supervised loss
                sup_loss = ce_loss(
                    outputs_weak[: args.labeled_bs], label_batch[: args.labeled_bs].long(),) + \
                    dice_loss(outputs_weak_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1)
                              )
                # complementary loss and adaptive sample weight for negative learning
                comp_loss, as_weight = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft, ce_loss=ce_loss)
                # unsupervised loss
                unsup_loss = (
                    ce_loss(outputs_strong[args.labeled_bs :], medsam_mask_ul.squeeze(1).long())
                    + dice_loss(outputs_strong_soft[args.labeled_bs :], medsam_mask_ul)
                    + as_weight * comp_loss
                )

                ## training samus
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
                loss_stu = sup_loss + consistency_weight * unsup_loss
                optimizer.zero_grad()
                loss_stu.backward()
                optimizer.step()
                ## update sam model
                optimizer_sam.zero_grad()
                loss_sam.backward()
                optimizer_sam.step()
                
                # update learning rate
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_  
                # update iteration number and save training log
                iter_num = iter_num + 1
                writer.add_scalar("lr", lr_, iter_num)
                writer.add_scalar("consistency_weight/consistency_weight", consistency_weight, iter_num)
                writer.add_scalar("loss/model_loss", loss_stu, iter_num)
                logging.info("iteration %d : model loss : %f" % (iter_num, loss_stu.item()))

                # update iteration number and save training log
                iter_num = iter_num + 1
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
                            classes=num_classes,
                            cuda_num=cuda_num,
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
                            snapshot_match_path,
                            "model_iter_{}_dice_{}.pth".format(iter_num, round(best_performance, 4)),
                        )
                        save_best = os.path.join(snapshot_match_path, "{}_best_model.pth".format(args.model))
                        util.save_checkpoint(epoch_num, matchnet, optimizer, loss_stu, save_mode_path)
                        util.save_checkpoint(epoch_num, matchnet, optimizer, loss_stu, save_best)
                        ## save sam
                        save_best_sam = os.path.join(snapshot_match_path, "best_sam_model.pth")
                        torch.save(sam.state_dict(), save_best_sam, _use_new_zipfile_serialization=False)

                    logging.info(
                        "iteration %d : model_mean_dice : %f model_mean_hd95 : %f" % (iter_num, performance, mean_hd95)
                    )
                    matchnet.train()

                if iter_num % 3000 == 0:
                    save_mode_path = os.path.join(snapshot_match_path, "model_iter_" + str(iter_num) + ".pth")
                    util.save_checkpoint(epoch_num, matchnet, optimizer, loss_stu, save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                
            ## Medsam: finish one epoch
            epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)            
            lr_scheduler_sam.step(epoch_loss_reduced)
            model_weights = sam.state_dict()
            checkpoint = {
                "model": model_weights,
                "epoch": epoch_num,
                "optimizer": optimizer.state_dict(),
                "loss": epoch_loss_reduced,
            }
            torch.save(checkpoint, os.path.join(snapshot_sam_path, "medsam_lite_latest_dual.pth"))
            if epoch_loss_reduced < best_loss:
                print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
                best_loss = epoch_loss_reduced
                checkpoint["best_loss"] = best_loss
                torch.save(checkpoint, os.path.join(snapshot_sam_path, "medsam_lite_best_dual.pth"))

            ## MatchNet: finish one epoch                






            










#         # load training samples
#         for i_batch, sampled_batch in enumerate(trainloader):
#             # load data
#             weak_batch, strong_batch, label_batch, label_bbox = (
#                 sampled_batch["image_weak"],
#                 sampled_batch["image_strong"],
#                 sampled_batch["label"], 
#                 sampled_batch["bbox"],
#             )
#             weak_batch, strong_batch, label_batch, label_bbox = (
#                 weak_batch.to(cuda_num),
#                 strong_batch.to(cuda_num),
#                 label_batch.to(cuda_num),
#                 label_bbox.to(cuda_num)
#             )

#             # outputs for model
#             outputs_weak = model(weak_batch) # without normalize
#             outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
#             outputs_strong = model(strong_batch)
#             outputs_strong_soft = torch.softmax(outputs_strong, dim=1)

#             # minmax normalization for softmax outputs before applying mask
#             pseudo_mask = (normalize(outputs_weak_soft) > args.conf_thresh).float()
#             outputs_weak_masked = outputs_weak_soft * pseudo_mask
#             pseudo_outputs = torch.argmax(outputs_weak_masked[args.labeled_bs :].detach(), dim=1, keepdim=False)
#             consistency_weight = get_current_consistency_weight(iter_num // 150)

#             if iter_num > args.iter_num:
#                 # SAM involved into training
#                 # sam_mask = sam_train()
#                 # for labeled data
#                 # generate mask from sam, Notice: normilize the input image: this code is ugly
#                 # because this data range is [0, 65535], astype(np.uint8) made all value lagrge that 255 to 255
#                 img_sam = weak_batch#[args.labeled_bs :]    
#                 img_sam = img_sam.detach().squeeze(1).permute(1,2,0).cpu().numpy()
#                 img_sam = TF.to_tensor(img_sam.astype(np.uint8)).unsqueeze(1).to(args.cuda_num)
#                 gt_sam = label_batch[: args.labeled_bs].long() # BHW
#                 # get 10 points from gt: 1 fg and 9 bg
#                 point_coord_gt, point_label_gt = find_point_label(gt_sam) # BHW
#                 # get 10 points from match-net pesudo-label: 1 fg and 9 bg
#                 pd_sam = outputs_weak_masked[args.labeled_bs:, 1,:,:] # BHW
#                 point_coord_pd, point_label_pd = find_point_label_pseudo(pd_sam)
#                 point_label = (torch.cat((point_coord_gt, point_coord_pd), dim=0),
#                                 torch.cat((point_label_gt, point_label_pd), dim=0))
#                 out_main = sam(img_sam, point_label, bbox=None) 
#                 predict = torch.sigmoid(out_main['masks']) # predict: B4HW
#                 sam_out_all = predict[:, 0, :, :] > 0.5  # (b, h, w)  
#                 sam_out = sam_out_all[args.labeled_bs:].long() # for unlabeled data as gt : BHW
#                 out_main = out_main['low_res_logits'][:,0,:,:].unsqueeze(1)#[:,:2,:,:] # use labeled data for training, B*4*H/2*W/2
                
#                 ## for match-net
#                 # supervised loss
#                 sup_loss = ce_loss(
#                     outputs_weak[: args.labeled_bs], label_batch[: args.labeled_bs].long(),) + \
#                     dice_loss(outputs_weak_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1)
#                               )

#                 # complementary loss and adaptive sample weight for negative learning
#                 comp_loss, as_weight = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft, ce_loss=ce_loss)
#                 # unsupervised loss
#                 unsup_loss = (
#                     ce_loss(outputs_strong[args.labeled_bs :], sam_out)
#                     + dice_loss(outputs_strong_soft[args.labeled_bs :], sam_out.unsqueeze(1))
#                     + as_weight * comp_loss
#                 )

#                 ## training samus
#                 teacher_pd = resize_pred(pseudo_outputs.unsqueeze(1), mode=InterpolationMode.NEAREST)
#                 loss_sam_unlabel = sam_dc_ce_loss(out_main[args.labeled_bs:, :,:,:], teacher_pd.float())
#                 ## for labeled data
#                 gt_label = resize_pred(label_batch[:][:args.labeled_bs].unsqueeze(1),  mode=InterpolationMode.NEAREST)
#                 loss_sam_label = sam_dc_ce_loss(out_main[:args.labeled_bs,:,:,:], gt_label.float())
#                 loss_sam = loss_sam_unlabel*0.5 + loss_sam_label

#                 ## update student model
#                 loss_stu = sup_loss + consistency_weight * unsup_loss
#                 optimizer.zero_grad()
#                 loss_stu.backward()
#                 optimizer.step()
#                 ## update sam model
#                 optimizer_sam.zero_grad()
#                 loss_sam.backward()
#                 optimizer_sam.step()
#             else: 
#                 # update student model only
#                 ## for match-net
#                 # supervised loss
#                 sup_loss = ce_loss(
#                     outputs_weak[: args.labeled_bs], label_batch[: args.labeled_bs].long(),) + \
#                     dice_loss(outputs_weak_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1)
#                               )

#                 # complementary loss and adaptive sample weight for negative learning
#                 comp_loss, as_weight = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft, ce_loss=ce_loss)
#                 # unsupervised loss
#                 unsup_loss = (
#                     ce_loss(outputs_strong[args.labeled_bs :], pseudo_outputs)
#                     + dice_loss(outputs_strong_soft[args.labeled_bs :], pseudo_outputs.unsqueeze(1))
#                     + as_weight * comp_loss
#                 )
#                 ## update student model
#                 loss_stu = sup_loss + consistency_weight * unsup_loss
#                 optimizer.zero_grad()
#                 loss_stu.backward()
#                 optimizer.step()                

#             # update ema model
#             update_ema_variables(model, ema_model, args.ema_decay, iter_num)
#             # update learning rate
#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group["lr"] = lr_


#             # update iteration number and save training log
#             iter_num = iter_num + 1
#             writer.add_scalar("lr", lr_, iter_num)
#             writer.add_scalar("consistency_weight/consistency_weight", consistency_weight, iter_num)
#             writer.add_scalar("loss/model_loss", loss_stu, iter_num)
#             logging.info("iteration %d : model loss : %f" % (iter_num, loss_stu.item()))
#             if iter_num % 50 == 0:
#                 image = weak_batch[1, 0:1, :, :]
#                 writer.add_image("train/Image", image, iter_num)
#                 outputs_weak = torch.argmax(torch.softmax(outputs_weak, dim=1), dim=1, keepdim=True)
#                 writer.add_image("train/model_Prediction", outputs_weak[1, ...] * 50, iter_num)

#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image("train/GroundTruth", labs, iter_num)

#             if iter_num > 0 and iter_num % 200 == 0:
#                 model.eval()
#                 metric_list = 0.0
#                 for i_batch, sampled_batch in enumerate(valloader):
#                     metric_i = test_single_volume(
#                         sampled_batch["image"],
#                         sampled_batch["label"], 
#                         sampled_batch["bbox"],
#                         model,
#                         classes=num_classes,
#                         cuda_num=cuda_num,
#                     )
#                     metric_list += np.array(metric_i)
#                 metric_list = metric_list / len(db_val)
#                 for class_i in range(num_classes - 1):
#                     writer.add_scalar(
#                         "info/model_val_{}_dice".format(class_i + 1),
#                         metric_list[class_i, 0],
#                         iter_num,
#                     )
#                     writer.add_scalar(
#                         "info/model_val_{}_hd95".format(class_i + 1),
#                         metric_list[class_i, 1],
#                         iter_num,
#                     )

#                 performance = np.mean(metric_list, axis=0)[0]

#                 mean_hd95 = np.mean(metric_list, axis=0)[1]
#                 writer.add_scalar("info/model_val_mean_dice", performance, iter_num)
#                 writer.add_scalar("info/model_val_mean_hd95", mean_hd95, iter_num)

#                 if performance > best_performance:
#                     best_performance = performance
#                     save_mode_path = os.path.join(
#                         snapshot_match_path,
#                         "model_iter_{}_dice_{}.pth".format(iter_num, round(best_performance, 4)),
#                     )
#                     save_best = os.path.join(snapshot_match_path, "{}_best_model.pth".format(args.model))
#                     util.save_checkpoint(epoch_num, model, optimizer, loss_stu, save_mode_path)
#                     util.save_checkpoint(epoch_num, model, optimizer, loss_stu, save_best)
#                     ## save sam
#                     save_best_sam = os.path.join(snapshot_match_path, "best_sam_model.pth")
#                     torch.save(sam.state_dict(), save_best_sam, _use_new_zipfile_serialization=False)

#                 logging.info(
#                     "iteration %d : model_mean_dice : %f model_mean_hd95 : %f" % (iter_num, performance, mean_hd95)
#                 )
#                 model.train()

#             if iter_num % 3000 == 0:
#                 save_mode_path = os.path.join(snapshot_match_path, "model_iter_" + str(iter_num) + ".pth")
#                 util.save_checkpoint(epoch_num, model, optimizer, loss_stu, save_mode_path)
#                 logging.info("save model to {}".format(save_mode_path))

#             if iter_num >= max_iterations:
#                 break
#             time1 = time.time()
#         if iter_num >= max_iterations:
#             iterator.close()
#             break
#     writer.close()


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