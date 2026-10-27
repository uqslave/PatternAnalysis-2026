import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from PIL import Image
from torchvision import transforms
from pathlib import Path
import math
from dataset import get_test_loader, LABELS, TEST_BATCH_SIZE
from modules import UNet, DiceLoss, dsc_per_class

def run_inference(model, loader, device, test_batch_size):
    # Four per row?
    fig_cum_cols = 4
    fig_cum_rows = int(math.ceil(len(loader) / fig_cum_cols))
    fig_cum, axes_cum = plt.subplots(fig_cum_rows, fig_cum_cols, figsize=(1540, 1540, "px"))
    fig_cum_diff, axes_cum_diff = plt.subplots(fig_cum_rows, fig_cum_cols, figsize=(1540, 1540, "px"))
    four_colour_cmap = mpl.colormaps['jet'].resampled(len(LABELS))
    dcs_scores = torch.Tensor().to(device)
    unet_report_test_temp = ""

    with torch.no_grad():
        model.eval()
        for bidx, (test_images, test_masks) in enumerate(loader):
            test_images = test_images.to(device)
            test_masks = test_masks.to(device)

            outputs = model(test_images)
            scores = dsc_per_class(outputs, test_masks)
            dcs_scores = torch.cat([dcs_scores, scores.reshape(-1, len(LABELS))], 0)

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

            # Generate coloured plot to highlight areas of greatest difference.
            # Blue -> dpx = 0, Red -> dpx = 1   Everything should be normalised to [0, 1] but idk how.
            # Just do abs error?
            diff_in_masks = test_masks[sample_no] - outputs[sample_no]

            for label_idx in range(len(LABELS)):
                test_mask_pixels = test_masks[sample_no][label_idx].cpu()
                reproduced_mask_pixels = outputs[sample_no][label_idx].cpu()
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

            fig_main.suptitle(f"Random Sample ({sample_no + 1}/{test_batch_size}) from batch {bidx + 1}", fontsize=14)
            fig_main.tight_layout()
            out_path = output_dir / f"rand_recon_sample_batch_{bidx + 1}.png"
            fig_main.savefig(out_path)

            diff_in_masks_cum = torch.sum(diff_in_masks, dim=0)
            cum_rows, cum_cols = int(bidx // fig_cum_cols), bidx % fig_cum_cols
            cum_sub_title = f"Cum Diff Batch {bidx + 1}"

            # Cumulative difference masks
            im_cum_diff = axes_cum_diff[cum_rows, cum_cols].imshow(diff_in_masks_cum.cpu().squeeze(), cmap="seismic",
                                                                   vmin=-1.0, vmax=1.0)
            fig_cum_diff.colorbar(im_cum_diff, orientation='vertical')
            axes_cum_diff[cum_rows, cum_cols].axis("off")
            axes_cum_diff[cum_rows, cum_cols].set_title(cum_sub_title, fontsize=12)

            # Cumulative mask
            cum_mask = torch.zeros_like(outputs[sample_no][0]).to(device)
            for class_idx in range(outputs[sample_no].shape[0]):
                class_mask = outputs[sample_no][class_idx]

                # Ceiling the class
                class_mask_ceil = torch.where(
                    class_mask > 0.5,
                    torch.ceil(class_mask) * class_idx,
                    class_mask
                )
                cum_mask = torch.add(cum_mask, class_mask_ceil)
            im_cum = axes_cum[cum_rows, cum_cols].imshow(cum_mask.cpu().squeeze(), cmap=four_colour_cmap, vmin=0.0,
                                                         vmax=len(LABELS))
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
    dcs_scores = torch.mean(dcs_scores, dim=0)
    for label, score in zip(LABELS, dcs_scores):
        unet_report_test_temp = unet_report_test_temp + f"{label}: {score:.3f}\n"
    unet_report_test_temp = unet_report_test_temp + f"\n"

    return unet_report_test_temp


test_dataloader = get_test_loader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
output_dir = Path("audit_out/")
output_dir.mkdir(parents=True, exist_ok=True)

model = UNet(in_channels=1, out_channels=len(LABELS), dropout_p=0.24)
model = model.to(device)

model_path = r"audit_out/unet_tce.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

inference_result = run_inference(model, test_dataloader, device, TEST_BATCH_SIZE)


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