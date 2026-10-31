import math
import random
import numpy as np
import dataset
import glob
import os
import torch
import pandas as pd
import matplotlib as mpl
from torch import Tensor
from matplotlib import pyplot as plt
from pathlib import Path

from dataset import SEED, LABELS


torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ======================================================================================================================
# =                                                 Reporting Helpers                                                  =
# ======================================================================================================================
def write_report(out_path: Path, metrics: tuple):
    """
    Write report to a txt file
    :param out_path: Path/dir to save the .txt report
    :param metrics: The metrics to write into the report
    """
    report = str()
    divider_length = 100

    for metric in metrics:
        report = report + metric
        report = report + "=" * divider_length + "\n\n"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    report_file = open(out_path, "w")
    report_file.write(report)
    report_file.close()


# ======================================================================================================================
# =                                                  Plotting Helpers                                                  =
# ======================================================================================================================
def plot_training(out_path: Path, t_loss: list[float], v_loss: list[float], t_score: list[float], v_score: list[float]):
    """
    Plot training loss vs validation loss and training dice scores vs validation dice scores.
    :param out_path: Path/dir to save the plot
    :param t_loss: Training losses
    :param v_loss: Validation losses
    :param t_score: Training dice scores
    :param v_score: Validation dice scores
    """

    # Plot training loss vs validation loss
    fig, ax1 = plt.subplots(figsize=(700, 700, "px"))
    ax1.plot(t_loss, color='tab:red', label='Training Loss')
    ax1.plot(v_loss, color='tab:orange', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Plot training dice score vs validation dice score
    ax2 = ax1.twinx()
    ax2.plot(t_score, color='tab:blue', label='Training DSC Score')
    ax2.plot(v_score, color='tab:green', label='Validation DSC Score')
    ax2.set_ylabel('DSC Score', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    fig.legend(loc="lower left")

    plt.title("U-Net Segmentation Training Curve")
    fig.tight_layout()

    fig.savefig(out_path)
    plt.show()


def plot_2d_inference(results: list[tuple[Tensor, Tensor]], device: torch.device, loader_len: int, batch_size: int):
    """
    Visualisation of the inference results for the 2D Improved U-Net segmentation model.
    Incl:
        predicted segmentation of a random slice from each batch,
        cumulative error between ground truth and prediction masks for each of the random slices.
        ground truth vs predicted masks for each of the classes for each of the random slice from each batch + error.
    :param results: Ground truth and predicted masks for each batch
    :param device: CUDA or CPU
    :param loader_len: length of the dataloader
    :param batch_size: Batch size
    """
    output_dir = Path("images/")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_cum_cols = 4
    fig_cum_rows = int(math.ceil(loader_len / fig_cum_cols))
    fig_cum, axes_cum = plt.subplots(fig_cum_rows, fig_cum_cols, figsize=(1540, 1540, "px"))
    fig_cum_diff, axes_cum_diff = plt.subplots(fig_cum_rows, fig_cum_cols, figsize=(1540, 1540, "px"))
    four_colour_cmap = mpl.colormaps['jet'].resampled(len(LABELS))

    for bidx, (test_masks, predictions) in enumerate(results):
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
        im_cum_diff = axes_cum_diff[cum_rows, cum_cols].imshow(diff_in_masks_cum.cpu().squeeze(), cmap="seismic", vmin=-1.0,
                                                               vmax=1.0)
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
        im_cum = axes_cum[cum_rows, cum_cols].imshow(cum_mask.cpu().squeeze(), cmap=four_colour_cmap, vmin=0.0,
                                                     vmax=len(LABELS))
        fig_cum.colorbar(im_cum, orientation='vertical')
        axes_cum[cum_rows, cum_cols].axis("off")
        axes_cum[cum_rows, cum_cols].set_title(cum_sub_title, fontsize=12)

    fig_cum_diff.suptitle("Cumulative Difference Between GND Truth and Reconstructed Masks", fontsize=18)
    out_path = output_dir / f"cum_diff_in_masks_{loader_len}_batches.png"
    fig_cum_diff.savefig(out_path)

    fig_cum.suptitle("MRI Masks Segmented", fontsize=18)
    out_path = output_dir / f"cum_masks_{loader_len}_batches.png"
    fig_cum.savefig(out_path)
    plt.show()
    plt.close()


# ======================================================================================================================
# =                                                   Helper Objects                                                   =
# ======================================================================================================================
class VRAMTracker:
    def __init__(self):
        self.peak_allocated = list()
        self.peak_reserved = list()

    def update(self):
        """
        Tracks peak allocated and peak reserved VRAMs
        Called after evaluating each batch.
        :return: None
        """
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # In MB
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)  # In MB

        self.peak_allocated.append(peak_allocated)
        self.peak_reserved.append(peak_reserved)

    def evaluate(self) -> tuple[float, float,float, float]:
        """
        Evaluate the max and mean VRAM statistics
        :return: Tuple of statistics
        """
        max_peak_allocated = max(self.peak_allocated)
        max_peak_reserved = max(self.peak_reserved)
        mean_peak_allocated = np.mean(self.peak_allocated)
        mean_peak_reserved = np.mean(self.peak_reserved)

        return max_peak_allocated, max_peak_reserved, mean_peak_allocated, mean_peak_reserved


class DataAudit:
    def __init__(self):
        self.image_files = sorted(glob.glob(os.path.join(dataset.DATASET_3D_TYPES["raw"], "*.nii*")))
        self.mask_files = sorted(glob.glob(os.path.join(dataset.DATASET_3D_TYPES["mask"], "*.nii*")))

        self.raw_scan_shapes        = list()
        self.mask_scan_shapes       = list()
        self.raw_mask_match         = list()
        self.mask_num_channels      = list()
        self.mask_channels_present  = list()
        self.patient_ids            = list()
        self.raw_file_names         = list()
        self.mask_file_names        = list()
        self.patient_count          = dict()
        self.to_csv = None

    def __repr__(self) -> str:
        if self.to_csv is None:
            raise "Data audit not yet evaluated!"
        out_str = f"Number of unique patients: {len(self.patient_count.keys())}\n"
        out_str = out_str + str(self.to_csv)
        return out_str

    def __str__(self) -> str:
        return self.__repr__()

    def evaluate(self):
        """
        Performs the data audit
        """
        assert len(self.image_files) == len(self.mask_files), "Data Mismatch!"

        patient_ids = list()
        raw_file_names, mask_file_names = list(), list()
        patient_count = dict()
        for index in range(len(self.image_files)):
            raw_file_name, mask_file_name = self.image_files[index].split("\\")[-1], self.mask_files[index].split("\\")[-1]
            raw_file_names.append(raw_file_name)
            mask_file_names.append(mask_file_name)
            patient_id = raw_file_name.split("_")[0]    # Extracts the patient id from the file name
            patient_ids.append(patient_id)
            patient_count[patient_id] = patient_count.get(patient_id, 0) + 1

        raw_scan_shapes = list()
        mask_scan_shapes = list()
        raw_mask_match = list()
        mask_num_channels = list()
        mask_channels_present = list()
        for patient_id in patient_count.keys():
            # shape for both tensors:
            #   raw shape   [H, W, D]
            #   mask shape  [H, W, D, K]
            for raw_scan, mask_scan in zip(dataset.RAW_COMPILED[patient_id], dataset.MASKS_COMPILED[patient_id]):
                raw_scan = torch.from_numpy(raw_scan)
                mask_scan = torch.from_numpy(mask_scan)

                raw_scan_shapes.append(raw_scan.shape)
                mask_labels = mask_scan.argmax(dim = 3)  # Also converts to [H, W, D]
                mask_scan_shapes.append(mask_labels.shape)

                mask_num_channels.append(mask_scan.shape[-1])
                labels = torch.unique(mask_labels)
                mask_channels_present.append(labels)

                raw_mask_match.append(raw_scan.shape == mask_labels.shape)


        assert (len(raw_file_names) == len(mask_file_names) ==
                len(raw_scan_shapes) == len(mask_scan_shapes) ==
                len(raw_mask_match) == len(mask_num_channels) == len(mask_channels_present)
            ), "Data Mismatch!"

        self.raw_scan_shapes        = raw_scan_shapes
        self.mask_scan_shapes       = mask_scan_shapes
        self.raw_mask_match         = raw_mask_match
        self.mask_num_channels      = mask_num_channels
        self.mask_channels_present  = mask_channels_present
        self.patient_ids            = patient_ids
        self.raw_file_names         = raw_file_names
        self.mask_file_names        = mask_file_names
        self.patient_count          = patient_count

        data = {
            "patient_id":               self.patient_ids,           # 0
            "raw_file_name":            self.raw_file_names,        # 1
            "mask_file_name":           self.mask_file_names,       # 2
            "raw_shape":                self.raw_scan_shapes,       # 3
            "mask_shape":               self.mask_scan_shapes,      # 4
            "shape_match":              self.raw_mask_match,        # 5
            "num_channels":             self.mask_num_channels,     # 6
            "mask_channels_present":    self.mask_channels_present, # 7
        }
        self.to_csv = pd.DataFrame(data)
        output_dir = Path("audit_out/")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.to_csv.to_csv(output_dir / "data_audit.csv", index=True)


# ======================================================================================================================
# =                                                     Main Guard                                                     =
# ======================================================================================================================
if __name__ == "__main__":
    audit = DataAudit()
    audit.evaluate()
    print(audit)