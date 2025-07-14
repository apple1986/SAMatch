from os import listdir, makedirs
from os.path import join, isfile, basename
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from networks.tiny_vit_sam import TinyViT
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import argparse
from collections import OrderedDict
import pandas as pd
from datetime import datetime
import h5py
from utils.get_prompts import find_box, find_centroid
from medpy import metric
import os
from scipy.ndimage.interpolation import zoom

#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-r',
    '--root_dir',
    type=str,
    default='/home/gxu/proj1/smatch/checkpoint/ACDC',
    # required=True,
    help='root directory of the code',
)

parser.add_argument(
    '--checkpoint_path',
    type=str,
    default="Unimatch_medsam_fix_debug_1_labeled_bs4/unet_drop",
    help='path to the checkpoint of MedSAM-Lite',
)

parser.add_argument(
    '--model_name',
    type=str,
    default="medsam_lite_best.pth",
    help='path to the model of MedSAM-Lite',
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default='predictions',
    help='directory to save the prediction',
)
parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default='./data/ACDC/volumes',
    # required=True,
    help='root directory of the data',
)

parser.add_argument(
    '-d',
    '--input_dir_list',
    type=str,
    default='./data/ACDC/test.list',
    # required=True,
    help='root directory of the data list',
)

# parser.add_argument(
#     '-s',
#     '--save_dir',
#     type=str,
#     default='./test_demo',
#     help='directory to save the prediction',
# )

parser.add_argument(
    '-device',
    type=str,
    default="cuda:1",
    help='device to run the inference',
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=1,
    help='number of workers for inference with multiprocessing',
)
parser.add_argument(
    '--save_overlay',
    default=True,
    help='whether to save the overlay image'
)
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='./test_demo/overlay',
    help='directory to save the overlay image'
)
args = parser.parse_args()

###############
data_root = args.input_dir
pred_save_dir = os.path.join(args.output_dir, args.checkpoint_path)
save_overlay = args.save_overlay
num_workers = args.num_workers
if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)

# checkpoint_path = os.path.join(args.root_dir ,args.checkpoint_path, "medsam", args.model_name)
checkpoint_path = os.path.join(args.root_dir ,args.checkpoint_path, args.model_name)
makedirs(pred_save_dir, exist_ok=True)
device = torch.device(args.device)
image_size = 256

###
def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

##########build model
class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks
    
### plot
def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))   

## process box
def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

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

## process box
def get_bbox256_ap(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    if mask_256.sum() > 0:
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
    else:
        bboxes256 = np.array([-100, -100, -100, -100])


    return bboxes256

def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
    """
    Perform inference using the LiteMedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box_256 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = None,
        boxes = box_torch,
        masks = None,
    )
    low_res_logits, iou = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)  
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg, iou

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

lite_medsam_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
medsam_lite_model.load_state_dict(lite_medsam_checkpoint)
# medsam_lite_model.load_state_dict(lite_medsam_checkpoint["model"])
medsam_lite_model.to(device)
medsam_lite_model.eval()

def MedSAM_infer_npz_2D(img_npz_file):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    ## preprocessing
    img_256 = resize_longest_side(img_3c, 256)
    newh, neww = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

    for idx, box in enumerate(boxes, start=1):
        box256 = resize_box_to_256(box, original_size=(H, W))
        box256 = box256[None, ...] # (1, 4)
        medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box256, (newh, neww), (H, W))
        segs[medsam_mask>0] = idx
        # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')
   
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )

    # visualize image, mask and bounding box
    if save_overlay:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box in enumerate(boxes):
            color = np.random.rand(3)
            box_viz = box
            show_box(box_viz, ax[1], edgecolor=color)
            show_mask((segs == i+1).astype(np.uint8), ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()

def MedSAM_infer_npz_3D(img_npz_file):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3D = npz_data['imgs'] # (D, H, W)
    spacing = npz_data['spacing'] # not used in this demo because it treats each slice independently
    segs = np.zeros_like(img_3D, dtype=np.uint8) 
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        z_max = min(z_max+1, img_3D.shape[0])
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c, 256)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)
            
            # convert the shape to (3, H, W)
            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)
            if z == z_middle:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            else:
                pre_seg = segs_3d_temp[z-1, :, :]
                pre_seg256 = resize_longest_side(pre_seg)
                if np.max(pre_seg256) > 0:
                    pre_seg256 = pad_image(pre_seg256)
                    box_256 = get_bbox256(pre_seg256)
                else:
                    box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        
        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        z_min = max(-1, z_min-1)
        for z in range(z_middle-1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)

            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)

            pre_seg = segs_3d_temp[z+1, :, :]
            pre_seg256 = resize_longest_side(pre_seg)
            if np.max(pre_seg256) > 0:
                pre_seg256 = pad_image(pre_seg256)
                box_256 = get_bbox256(pre_seg256)
            else:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        segs[segs_3d_temp>0] = idx
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )            

    # visualize image, mask and bounding box
    if save_overlay:
        idx = int(segs.shape[0] / 2)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[1].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box3D in enumerate(boxes_3D, start=1):
            if np.sum(segs[idx]==i) > 0:
                color = np.random.rand(3)
                x_min, y_min, z_min, x_max, y_max, z_max = box3D
                box_viz = np.array([x_min, y_min, x_max, y_max])
                show_box(box_viz, ax[1], edgecolor=color)
                show_mask(segs[idx]==i, ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()

#############metrics
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd

def resize_val(image):
    b,x, y = image.shape
    return zoom(image, (1, 256 / x, 256 / y), order=0) # slice_Num x H x W
##################### load ACDC data
# get file path
file_folder = args.input_dir_list
with open(file_folder, 'r') as f:
        image_list = f.readlines()
print(len(image_list))
print(image_list[0])
image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
print(image_list)
# load data
first_total = 0.0
second_total = 0.0
third_total = 0.0
for case in tqdm(image_list):
    h5f = h5py.File(args.input_dir + "/{}.h5".format(case), 'r')
    image_vol = h5f['image'][:]
    label_vol = h5f['label'][:]
    image_vol = resize_val(image_vol)
    label_vol = resize_val(label_vol)
    # print(image_vol.shape)
    # print(label_vol.shape)
    # normalize image to [0, 1]
    img_vol_256 = (image_vol - image_vol.min()) / (image_vol.max()- image_vol.min()+1e-9) # Notice: normalize
    H, W = img_vol_256.shape[1:]
    pd_vol = np.zeros((label_vol.shape[0], 256, 256))
    medsam_logit_all = torch.zeros((1, 4, 256, 256))
    ## for each slice
    for idx in range(0, img_vol_256.shape[0]):
        img_256 = img_vol_256[idx, :,:]
        # img_256 = resize_longest_side(img_256)
        img_256_3c = np.stack((img_256, img_256, img_256)) # 3HW


        img_256_tensor = torch.tensor(img_256_3c).float().unsqueeze(0).to(device)
        gt_256 = label_vol[idx, :,:]
        # gt_256 = resize_longest_side(gt_256)
        # get image embedding
        with torch.no_grad():
            # image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
            # print(image_embedding.shape) # Bx256x64x64
            ## find box from GT mask
            if gt_256.sum() > 0:
                for n in range(1, 4):
                    box256 = get_bbox256_ap(gt_256==n, bbox_shift=3)
                    box256 = box256[None, ...] # (1,4)
                    box256 = torch.tensor(box256).unsqueeze(0).to(device)
                    medsam_logit = medsam_lite_model(img_256_tensor, box256)
                    medsam_logit = torch.sigmoid(medsam_logit)
                    medsam_logit_all[:, n, :, :] = medsam_logit[:,0,:,:]
                medsam_logit_all[:, 0, :, :] = (1 - torch.mean(medsam_logit_all[:, 1:, :, :], dim=1, keepdim=True)).squeeze(1) # for the background
                pred_w_sam = medsam_logit_all.softmax(dim=1)
                medsam_mask = pred_w_sam.argmax(dim=1)

                # box256 = get_bbox256(gt_256, bbox_shift=3)
                # box256 = box256[None, ...] # (1,4)
                # ## inference
                # medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box256, (H, W), (H, W))
                # # print(f'{case}_{idx}, box: {box256}, predicted iou: {np.round(iou_pred.item(), 4)}')
                pd_vol[idx, :,:] = medsam_mask
            else:
                pd_vol[idx, :,:] = gt_256

    ## cal 3D dice
    # first_metric = calculate_metric_percase(pd_vol == 1, label_vol == 1)
    # print("%.4f\t%.3f\t%.3f" % (first_metric[0], first_metric[1], first_metric[2]))
    # first_total += np.asarray(first_metric)

    first_metric = calculate_metric_percase(pd_vol == 1, label_vol == 1) # RV
    second_metric = calculate_metric_percase(pd_vol == 2, label_vol == 2) # MYO
    third_metric = calculate_metric_percase(pd_vol == 3, label_vol == 3) # LV
    first_total += np.asarray(first_metric)
    second_total += np.asarray(second_metric)
    third_total += np.asarray(third_metric)
    print(f"case: {case}, dice: {first_metric}")


metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
print(metric)
# save_res_path = "{}/test_metric_medsamLit_box_shift3.txt".format(args.save_dir)
save_res_path = os.path.join(args.root_dir, args.checkpoint_path, "metric_medsamLit_box_shift3_gt.txt")
# with open(save_res_path,'w') as f:
#     res = "%.4f\t%.3f\t%.3f" % (avg_metric[0], avg_metric[1], avg_metric[2])
#     f.write(res)
mean_metric = np.array((metric[0]+metric[1]+metric[2])/3)
# save_res_path = "./checkpoint/ACDC/{}/test_metrics.txt".format(args.checkpoint_path)
with open(save_res_path,'w') as f:
    for n in range(3):
        res = "%.4f\t%.3f\t%.3f\n" % (metric[n][0], metric[n][1], metric[n][2])
        f.write(res)
    res_dice = "Dice RV, MYO and LV:\n%.4f\t%.4f\t%.4f\n" % (metric[0][0], metric[1][0], metric[2][0])
    f.write(res_dice)        
    res = "Mean Dice, HD95 and ASD\n%.4f\t%.3f\t%.3f\n" % (mean_metric[0], mean_metric[1], mean_metric[2])
    f.write(res)

# if __name__ == '__main__':
#     img_npz_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
#     efficiency = OrderedDict()
#     efficiency['case'] = []
#     efficiency['time'] = []
#     for img_npz_file in tqdm(img_npz_files):
#         start_time = time()
#         if basename(img_npz_file).startswith('3D'):
#             MedSAM_infer_npz_3D(img_npz_file)
#         else:
#             MedSAM_infer_npz_2D(img_npz_file)
#         end_time = time()
#         efficiency['case'].append(basename(img_npz_file))
#         efficiency['time'].append(end_time - start_time)
#         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
#     efficiency_df = pd.DataFrame(efficiency)
#     efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)