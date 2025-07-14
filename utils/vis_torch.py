import torch
from kornia.utils import tensor_to_image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.contrib.models import SegmentationResults

def colorize_masks(binary_masks: torch.Tensor, merge: bool = True, alpha: None | float = None) -> list[torch.Tensor]:
    """Convert binary masks (B, C, H, W), boolean tensors, into masks with colors (B, (3, 4) , H, W) - RGB or RGBA. Where C refers to the number of masks.
    Args:
        binary_masks: a batched boolean tensor (B, C, H, W)
        merge: If true, will join the batch dimension into a unique mask.
        alpha: alpha channel value. If None, will generate RGB images

    Returns:
        A list of `C` colored masks.
    """
    B, C, H, W = binary_masks.shape
    OUT_C = 4 if alpha else 3

    output_masks = []

    for idx in range(C):
        _out = torch.zeros(B, OUT_C, H, W, device=binary_masks.device, dtype=torch.float32)
        for b in range(B):
            color = torch.rand(1, 3, 1, 1, device=binary_masks.device, dtype=torch.float32)
            if alpha:
                color = torch.cat([color, torch.tensor([[[[alpha]]]], device=binary_masks.device, dtype=torch.float32)], dim=1)

            to_colorize = binary_masks[b, idx, ...].view(1, 1, H, W).repeat(1, OUT_C, 1, 1)
            _out[b, ...] = torch.where(to_colorize, color, _out[b, ...])
        output_masks.append(_out)

    if merge:
        output_masks = [c.max(dim=0)[0] for c in output_masks]

    return output_masks


def show_binary_masks(binary_masks: torch.Tensor, axes) -> None:
    """plot binary masks, with shape (B, C, H, W), where C refers to the number of masks.

    will merge the `B` channel into a unique mask.
    Args:
        binary_masks: a batched boolean tensor (B, C, H, W)
        ax: a list of matplotlib axes with lenght of C
    """
    colored_masks = colorize_masks(binary_masks, True, 0.6)

    for ax, mask in zip(axes, colored_masks):
        ax.imshow(tensor_to_image(mask))


def show_boxes(boxes: Boxes, ax) -> None:
    boxes_tensor = boxes.to_tensor(mode="xywh").detach().cpu().numpy()
    for box in boxes_tensor:
        x0, y0, w, h = box
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="orange", facecolor=(0, 0, 0, 0), lw=2))


def show_points(points: tuple[Keypoints, torch.Tensor], ax, marker_size=200):
    coords, labels = points
    pos_points = coords[labels == 1].to_tensor().detach().cpu().numpy()
    neg_points = coords[labels == 0].to_tensor().detach().cpu().numpy()

    ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="+", s=marker_size, linewidth=2)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="x", s=marker_size, linewidth=2)


def show_image(image: torch.Tensor):
    plt.imshow(tensor_to_image(image))
    plt.axis("off")
    plt.show()


def show_predictions(
    image: torch.Tensor,
    predictions: SegmentationResults,
    points: tuple[Keypoints, torch.Tensor] | None = None,
    boxes: Boxes | None = None,
) -> None:
    n_masks = predictions.logits.shape[1]

    fig, axes = plt.subplots(1, n_masks, figsize=(21, 16))
    axes = [axes] if n_masks == 1 else axes

    for idx, ax in enumerate(axes):
        score = predictions.scores[:, idx, ...].mean()
        ax.imshow(tensor_to_image(image))
        ax.set_title(f"Mask {idx+1}, Score: {score:.3f}", fontsize=18)

        if points:
            show_points(points, ax)

        if boxes:
            show_boxes(boxes, ax)

        ax.axis("off")

    show_binary_masks(predictions.binary_masks, axes)
    plt.show()