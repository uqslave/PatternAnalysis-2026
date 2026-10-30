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

from torch.utils.data import DataLoader

from dataset import LABELS, BATCH_SIZE_2D, get_dataloader_2d, get_dataloader_3d, BATCH_SIZE_3D
from modules import dsc_per_class, ImprovedUnet, ImprovedUnet3D, iou_per_class, fpr_per_class

# ======================================================================================================================
# =                                               Score Summary Helpers                                                =
# ======================================================================================================================
@torch.no_grad()
def summarise_scores(predictions, targets):
    dsc_scores = torch.Tensor().to(device)

    predictions = torch.argmax(predictions, dim=1)  # Becomes [B, H, W]
    predictions = F.one_hot(predictions, num_classes=len(LABELS)).permute(0, 3, 1, 2).float()  # Back to [B, K, H, W]

    # Dice +empty slices
    dsc_wempty = dsc_per_class(predictions, targets)  # [B, K]

    # Dice present slices
    B, K = predictions.shape[:2]
    tp = targets > 0
    # 1. Find where each class is present in each batch item -> shape: [B, K]
    has_class = tp.reshape(B, K, -1).any(dim=2)
    # 2. Expand `has_class` back to match the spatial shape [B, K, H, W]
    # (Assuming H, W are dimensions 2 and 3)
    spatial_dims = predictions.shape[2:]
    mask_expanded = has_class.view(B, K, *([1] * len(spatial_dims)))
    # 3. Filter predictions: zero out everything where the class isn't present
    filtered_predictions = predictions * mask_expanded.float()
    wtf = targets > 0

    # Dice pooled slices
    huh = predictions[:, 5, :, :].unsqueeze(dim=1)
    umm = targets[:, 5, :, :].unsqueeze(dim=1)
    dsc_pooled = dsc_per_class(huh, umm)

    # IoU pooled
    iou = iou_per_class(huh, umm)

    # Empty FP rate
    fpr = fpr_per_class(predictions, targets)

    return dsc_wempty, dsc_pooled, iou, fpr

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
    dsc_scores1, dsc_scores2, dsc_scores3, dsc_scores4 = torch.Tensor().to(device),torch.Tensor().to(device),torch.Tensor().to(device),torch.Tensor().to(device)
    unet_report_test_temp = ""


    model.eval()
    for bidx, (test_images, test_masks) in enumerate(loader):
        test_images = test_images.to(device)
        test_masks = test_masks.to(device)

        predictions = model(test_images)

        scores, dsc_pooled, iou, fpr = summarise_scores(predictions, test_masks)

        dsc_scores1 = torch.cat([dsc_scores1, scores], 0)
        dsc_scores2 = torch.cat([dsc_scores2, dsc_pooled], 0)
        dsc_scores3 = torch.cat([dsc_scores3, iou], 0)
        dsc_scores4 = torch.cat([dsc_scores4, fpr], 0)

        # scores = dsc_per_class(predictions, test_masks)  # [B, K]
        scores = torch.mean(scores, dim=0)
        print(f"Test batch {bidx + 1}/{len(loader)} DSC Scores: ")
        for name, s in zip(LABELS, scores):
            print(f"{name}: {s:.4f}")
        print("")

        unet_report_test_temp = unet_report_test_temp + f"Mean DSC per class for test set batch {bidx + 1}/{len(loader)}\n"
        for label, score in zip(LABELS, scores):
            unet_report_test_temp = unet_report_test_temp + f"{label}: {score:.3f}\n"
        unet_report_test_temp = unet_report_test_temp + f"\n"

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

    unet_report_test_temp = unet_report_test_temp + f"--- Mean DSC per class across test set ---\n"
    dsc_scores1 = torch.mean(dsc_scores1, dim=0)
    dsc_scores2 = torch.mean(dsc_scores2, dim=0)
    dsc_scores3 = torch.mean(dsc_scores3, dim=0)
    dsc_scores4 = torch.mean(dsc_scores4, dim=0)
    print(dsc_scores1, "\n", dsc_scores2, "\n", dsc_scores3, "\n", dsc_scores4)
    for label, score in zip(LABELS, dsc_scores1):
        unet_report_test_temp = unet_report_test_temp + f"{label}: {score:.3f}\n"
    unet_report_test_temp = unet_report_test_temp + f"\n"

    return unet_report_test_temp


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

    # 1. Training set per epoch
    # 2. Validation set per epoch
    # 3. Testing set per batch and mean across set

    unet_report_str = unet_report_str + inference_result
    unet_report_str = unet_report_str + "=" * divider_length + "\n\n"
    out_path = output_dir / "inference_results.txt"
    unet_report_file = open(out_path, "w")
    unet_report_file.write(unet_report_str)
    unet_report_file.close()