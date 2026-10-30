"""
How to run:
    python predict.py --model [3D | 2D]
Choose 3D to run inference with 3D Improved UNet;
Choose 2D to run inference with 2D Improved UNet;
"""

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn.functional as F
from pathlib import Path
import math
import argparse

from torch import Tensor
from torch.utils.data import DataLoader
from dataset import LABELS, BATCH_SIZE_2D, get_dataloader_2d, get_dataloader_3d, BATCH_SIZE_3D
from modules import dsc_per_class, ImprovedUnet, ImprovedUnet3D, iou_per_class, fpr_per_class
from utils import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)

# ======================================================================================================================
# =                                               Score Summary Helpers                                                =
# ======================================================================================================================
def _one_hot(labels: Tensor, K: int) -> Tensor:
    """
    Converts [N, (D), H, W] to [N, K, (D), H, W]
    :param labels: [N, (D), H, W] labels
    :param K: Number of classes
    :return: Mask with one-hot encoding
    """
    return F.one_hot(labels.long(), K).movedim(-1, 1).float()


@torch.no_grad()
def compute_metrics(predictions: Tensor, targets: Tensor, K: int = len(LABELS)) -> dict[str, Tensor]:
    """
    Label maps [N, H, W] (2D) or [N, D, H, W] (3D). Called after processing every batch.
    Evaluates:
        DSC (+ empty),\n
        DSC (present),\n
        DSC (pooled),\n
        IoU (pooled),\n
        FPR (pooled),\n
    Every value returned has shape [K].
    :param predictions: Predictions
    :param targets: Targets (Ground Truth)
    :param K: Number of classes
    :return: Evaluation metrics
    """
    pred_onehot, targ_onehot = _one_hot(predictions, K), _one_hot(targets, K)
    present = targ_onehot.flatten(2).sum(2) > 0     # [N, K] class exists in the ground truth

    dsc = dsc_per_class(pred_onehot, targ_onehot)   # [N, K] per slice
    dsc_wempty = dsc.mean(0)
    dsc_present = (dsc * present).sum(0) / present.sum(0).clamp(min=1)

    # Change [N, K, dims] to [K, N, dims]
    # Unsqueeze to make it [1, K, N, dims]. Now can be treated as a tensor with batch size of 1.
    pred_onehot = pred_onehot.movedim(0, 1).unsqueeze(0)
    targ_onehot = targ_onehot.movedim(0, 1).unsqueeze(0)

    dsc_pooled = dsc_per_class(pred_onehot, targ_onehot)[0]
    iou_pooled = iou_per_class(pred_onehot, targ_onehot)[0]
    pixel_fpr = fpr_per_class(pred_onehot, targ_onehot)[0]

    return {
        "n_present"         : present.sum(0),
        "dsc_wempty"        : dsc_wempty,
        "dsc_present"       : dsc_present,
        "dsc_pooled"        : dsc_pooled,
        "iou_pooled"        : iou_pooled,
        "pixel_fpr"         : pixel_fpr
    }


def metrics_to_string(metrics: dict) -> str:
    lines = [
        f"{'class':<13}"
        f"{'#present':>9}"
        f"{'Dice(+empty)':>14}"
        f"{'Dice(present)':>15}"
        f"{'Dice(pooled)':>14}"
        f"{'IoU(pooled)':>13}"
        f"{'pixelFPR':>10}"
    ]
    for k, name in enumerate(LABELS):
        lines.append(
            f"{name:<13}"
            f"{int(metrics['n_present'][k]):>9}"
            f"{metrics['dsc_wempty'][k]:>14.3f}"
            f"{metrics['dsc_present'][k]:>15.3f}"
            f"{metrics['dsc_pooled'][k]:>14.3f}"
            f"{metrics['iou_pooled'][k]:>13.3f}"
            f"{metrics['pixel_fpr'][k]:>10.4f}")
    return "\n".join(lines)


# ======================================================================================================================
# =                                               Inference (Test) Loop                                                =
# ======================================================================================================================
@torch.no_grad()
def run_inference2d(model: ImprovedUnet, loader: DataLoader, device: torch.device, batch_size: int) -> str:
    """
    Run inference with 2D Improved UNet
    :param model: 2D Improved UNet
    :param loader: Test set dataloader
    :param device: CUDA or CPU
    :param batch_size: Test batch size
    :return: Inference results
    """
    fig_cum_cols = 4
    fig_cum_rows = int(math.ceil(len(loader) / fig_cum_cols))
    fig_cum, axes_cum = plt.subplots(fig_cum_rows, fig_cum_cols, figsize=(1540, 1540, "px"))
    fig_cum_diff, axes_cum_diff = plt.subplots(fig_cum_rows, fig_cum_cols, figsize=(1540, 1540, "px"))
    four_colour_cmap = mpl.colormaps['jet'].resampled(len(LABELS))
    predictions_all, targets_all = list(), list()

    model.eval()
    for bidx, (test_images, test_masks) in enumerate(loader):
        test_images = test_images.to(device)
        test_masks = test_masks.to(device)
        predictions = model(test_images)

        predictions_all.append(predictions.argmax(1).to(torch.uint8).cpu())
        targets_all.append(test_masks.argmax(1).to(torch.uint8).cpu())

        fig_main, axes_main = plt.subplots(3, len(LABELS), figsize=(1440, 900, "px"))
        sample_no = random.randint(0, len(test_masks) - 1)

        diff_in_masks = test_masks[sample_no] - predictions[sample_no]

        for label_idx in range(len(LABELS)):
            test_mask_pixels = test_masks[sample_no][label_idx].cpu()
            reproduced_mask_pixels = predictions[sample_no][label_idx].cpu()
            diff_in_mask_pixels = diff_in_masks[label_idx].cpu()

            axes_main[0, label_idx].set_title(LABELS[label_idx], fontsize=12)
            # Ground truth mask
            axes_main[0, label_idx].imshow(test_mask_pixels.squeeze(), cmap="grey")
            axes_main[0, label_idx].axis("off")

            # UNET output mask
            axes_main[1, label_idx].imshow(reproduced_mask_pixels.squeeze(), cmap="grey")
            axes_main[1, label_idx].axis("off")

            # Difference in mask
            im = axes_main[2, label_idx].imshow(diff_in_mask_pixels.squeeze(), cmap="seismic", vmin=-1.0, vmax=1.0)
            fig_main.colorbar(im, orientation='vertical')
            axes_main[2, label_idx].axis("off")
            fig_main.supylabel('Difference        |        Reconstructed        |        Ground Truth', ha='left',
                               va='center', fontsize=18)

        fig_main.suptitle(f"Random Sample ({sample_no + 1}/{batch_size}) from batch {bidx + 1}", fontsize=14)
        fig_main.tight_layout()
        out_path = output_dir / f"rand_recon_sample_batch_{bidx + 1}.png"
        fig_main.savefig(out_path)

        diff_in_masks_cum = torch.sum(diff_in_masks, dim=0)
        cum_rows, cum_cols = int(bidx // fig_cum_cols), bidx % fig_cum_cols
        cum_sub_title = f"Cum Diff Batch {bidx + 1}"

        # Cumulative difference masks
        im_cum_diff = axes_cum_diff[cum_rows, cum_cols].imshow(diff_in_masks_cum.cpu().squeeze(), cmap="seismic", vmin=-1.0, vmax=1.0)
        fig_cum_diff.colorbar(im_cum_diff, orientation='vertical')
        axes_cum_diff[cum_rows, cum_cols].axis("off")
        axes_cum_diff[cum_rows, cum_cols].set_title(cum_sub_title, fontsize=12)

        # Cumulative mask
        cum_mask = torch.zeros_like(predictions[sample_no][0]).to(device)
        for class_idx in range(predictions[sample_no].shape[0]):
            class_mask = predictions[sample_no][class_idx]

            # Ceiling the class
            class_mask_ceil = torch.where(
                class_mask > 0.5,
                torch.ceil(class_mask) * class_idx,
                class_mask
            )
            cum_mask = torch.add(cum_mask, class_mask_ceil)
        im_cum = axes_cum[cum_rows, cum_cols].imshow(cum_mask.cpu().squeeze(), cmap=four_colour_cmap, vmin=0.0, vmax=len(LABELS))
        fig_cum.colorbar(im_cum, orientation='vertical')
        axes_cum[cum_rows, cum_cols].axis("off")
        axes_cum[cum_rows, cum_cols].set_title(cum_sub_title, fontsize=12)

    fig_cum_diff.suptitle("Cumulative Difference Between GND Truth and Reconstructed Masks", fontsize=18)
    out_path = output_dir / f"cum_diff_in_masks_{len(loader)}_batches.png"
    fig_cum_diff.savefig(out_path)

    fig_cum.suptitle("MRI Masks Segmented", fontsize=18)
    out_path = output_dir / f"cum_masks_{len(loader)}_batches.png"
    fig_cum.savefig(out_path)

    plt.show()

    table = metrics_to_string(compute_metrics(torch.cat(predictions_all), torch.cat(targets_all)))
    print(table)

    return "\n" + table + "\n"


@torch.no_grad()
def run_inference3d(model: ImprovedUnet3D, loader: DataLoader, device: torch.device, batch_size: int) -> str:
    """
    Run inference with 3D Improved UNet
    :param model: 3D Improved UNet
    :param loader: Test set dataloader
    :param device: CUDA or CPU
    :param batch_size: Test batch size
    :return: Inference results
    """
    model.eval()

    return ""


# ======================================================================================================================
# =                                                     Main Guard                                                     =
# ======================================================================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = Path("audit_out/")
    output_dir.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type = str, required = True)
    args = ap.parse_args()

    if args.model == "3d":
        # For 3D Improved UNet
        model = ImprovedUnet3D(in_channels=1, out_channels=len(LABELS))
        model = model.to(device)
        model_path = r"audit_out/unet_model_3D.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_dataloader = get_dataloader_3d("test")
        inference_result = run_inference3d(model, test_dataloader, device, BATCH_SIZE_3D["test"])
    elif args.model == "2d":
        # For 2D Improved UNet
        model = ImprovedUnet(in_channels=1, out_channels=len(LABELS))
        model = model.to(device)

        model_path = r"audit_out/unet_improved_dsc.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))

        test_dataloader = get_dataloader_2d("test")
        inference_result = run_inference2d(model, test_dataloader, device, BATCH_SIZE_2D["test"])
    else:
        raise Exception("Unknown model")

    # Write to file
    unet_report_str = str()
    divider_length = 100

    unet_report_str = unet_report_str + inference_result
    unet_report_str = unet_report_str + "=" * divider_length + "\n\n"
    out_path = output_dir / "inference_results.txt"
    unet_report_file = open(out_path, "w")
    unet_report_file.write(unet_report_str)
    unet_report_file.close()