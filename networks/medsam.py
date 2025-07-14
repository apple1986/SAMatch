import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
### medsam
from model_sam.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from networks.tiny_vit_sam import TinyViT
## loss
import monai



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
def load_medsam(ckpt_sam, cuda_num="cuda:0"):
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
    lite_medsam_checkpoint = torch.load(ckpt_sam, map_location='cpu', weights_only=True)

    if 'unet_drop/medsam_lite_best.pth' in ckpt_sam:
        medsam_lite_model.load_state_dict(lite_medsam_checkpoint)
    elif "medsam_lite_best.pth" in ckpt_sam:
        medsam_lite_model.load_state_dict(lite_medsam_checkpoint["model"])
    else:    
        medsam_lite_model.load_state_dict(lite_medsam_checkpoint, strict=True)
    medsam_lite_model.to(cuda_num)
    # %
    print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
    # %
    return medsam_lite_model

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

def sam_set_uni(medsam_lite_model, lr=5e-5):
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

    return optimizer, lr_scheduler
