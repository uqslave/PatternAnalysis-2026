import os
import numpy as np
import torch
import random
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import LABELS, get_dataloader_2d, get_dataloader_3d
from modules import dsc_per_class, DiceCELoss, ImprovedUnet, ImprovedUnet3D
from predict import metrics_to_string, compute_metrics
from utils import SEED

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)


# ======================================================================================================================
# =                                             Hyperparameter definitions                                             =
# ======================================================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channels = 1 # Greyscale
out_channels = len(LABELS)
dropout = 0.25
model = ImprovedUnet3D(in_channels, out_channels, dropout)
model = model.to(device)

num_epochs = 60
learning_rate = 3.1e-4
optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_epochs, eta_min=1.2e-7)
criterion = DiceCELoss(w=0.5)
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

output_dir = Path("audit_out/")
output_dir.mkdir(parents=True, exist_ok=True)


def main():
    print("Using device:", device.type)
    train_dataloader = get_dataloader_3d("train")
    validate_dataloader = get_dataloader_3d("validate")

    training_losses = list()
    validation_losses = list()
    training_scores = list()
    validation_scores = list()
    unet_report_path = output_dir / r"unet_report.txt"
    predictions_all, targets_all = list(), list()

    unet_report_train_temp = str()
    unet_report_validation_temp = str()

    print("Training...")
    start_time = time.time()
    training_times = list()
    training_subtime_cum = start_time
    for epoch in range(num_epochs):
        # ==============================================================================================================
        # =                                               Training Loop                                                =
        # ==============================================================================================================
        training_subtime = time.time()
        model.train()
        epoch_loss = validation_loss = 0
        losses_per_batch = torch.Tensor().to(device)
        losses_per_batch_temp = torch.Tensor().to(device)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (images, masks) in enumerate(train_dataloader):
            #print(f"\tBatch {batch_idx + 1}/{len(train_dataloader)}")
            images = images.to(device)
            masks = masks.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()

            losses_per_batch = torch.cat(
                [
                    losses_per_batch,
                    torch.mean(dsc_per_class(outputs, masks), dim=0).reshape(-1, len(LABELS))
                ]
            )
            epoch_loss = epoch_loss + loss.item()

        unet_report_train_temp = unet_report_train_temp + f"Mean DSC per class at epoch {epoch + 1}/{num_epochs}\n"
        losses_per_batch = torch.mean(losses_per_batch, dim=0)
        for label_idx, label in enumerate(LABELS):
            unet_report_train_temp = unet_report_train_temp + f"{label}: {losses_per_batch[label_idx]:.3f}\n"
        unet_report_train_temp = unet_report_train_temp + f"\n"

        avg_loss = epoch_loss / len(train_dataloader)
        training_losses.append(avg_loss)
        training_scores.append(1 - avg_loss)
        out_str = f"\tAverage Loss: {avg_loss:.3f}"

        training_subtime_cum = time.time() - training_subtime_cum
        training_subtime = time.time() - training_subtime
        training_times.append(training_subtime)

        # ==============================================================================================================
        # =                                              Validation Loop                                               =
        # ==============================================================================================================
        #print("\tPerforming Validation...")
        model.eval()
        with torch.no_grad():
            for batch_idx, (val_images, val_masks) in enumerate(validate_dataloader):
                print(f"\tBatch {batch_idx + 1}/{len(validate_dataloader)}")
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)

                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(val_images)
                    val_loss = criterion(outputs, val_masks)
                validation_loss = validation_loss + val_loss.item()
                losses_per_batch_temp = torch.cat(
                    [
                        losses_per_batch_temp,
                        torch.mean(dsc_per_class(outputs, val_masks), dim=0).reshape(-1, len(LABELS))
                    ]
                )
                predictions_all.append(outputs.argmax(1).to(torch.uint8).cpu())
                targets_all.append(val_masks.argmax(1).to(torch.uint8).cpu())

            unet_report_validation_temp = unet_report_validation_temp + (f"Mean DSC per class for validation set at "
                                                                         f"epoch {epoch + 1}/{num_epochs}\n")
            losses_per_batch_temp = torch.mean(losses_per_batch_temp, dim=0)
            for label_idx, label in enumerate(LABELS):
                unet_report_validation_temp = unet_report_validation_temp + (f"{label}: "
                                                                             f"{losses_per_batch_temp[label_idx]:.3f}\n")
            unet_report_validation_temp = unet_report_validation_temp + f"\n"
        scheduler.step()
        avg_validation_loss = validation_loss / len(validate_dataloader)
        validation_losses.append(avg_validation_loss)
        validation_scores.append(1 - avg_validation_loss)
        out_str = out_str + (f" | Validation Loss: {avg_validation_loss:.3f}\t"
                             f"Time: {training_subtime:.3f}s\tTime Cum: {training_subtime_cum}")
        training_subtime_cum = time.time() - training_subtime_cum
        print(out_str)

        del outputs, loss
        torch.cuda.empty_cache()

    # ==================================================================================================================
    # =                                          Save model & Training Plots                                           =
    # ==================================================================================================================
    outpath = output_dir / f"unet_improved_3d.pth"
    torch.save(model.state_dict(), outpath)
    print(f"Saved checkpoint to {outpath}")
    print("")

    # Plot training loss vs validation loss
    fig, ax1 = plt.subplots(figsize=(700, 700, "px"))
    ax1.plot(training_losses, color='tab:red', label='Training Loss')
    ax1.plot(validation_losses, color='tab:orange', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.plot(training_scores, color='tab:blue', label='Training DSC Score')
    ax2.plot(validation_scores, color='tab:green', label='Validation DSC Score')
    ax2.set_ylabel('DSC Score', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    fig.legend(loc="lower left")

    plt.title("UNET Segmentation Training Curve")
    fig.tight_layout()
    outpath = output_dir / r"3D UNET Segmentation Training Curve.png"
    fig.savefig(outpath)
    plt.show()

    table = metrics_to_string(compute_metrics(torch.cat(predictions_all), torch.cat(targets_all)))

    # Write to file
    unet_report_str = str()
    divider_length = 100

    # 1. Training set per epoch
    # 2. Validation set per epoch
    # 3. Testing set per batch and mean across set

    unet_report_str = unet_report_str + unet_report_train_temp
    unet_report_str = unet_report_str + "=" * divider_length + "\n\n"

    unet_report_str = unet_report_str + unet_report_validation_temp
    unet_report_str = unet_report_str + "=" * divider_length + "\n\n"

    unet_report_str = unet_report_str + table
    unet_report_str = unet_report_str + "=" * divider_length + "\n\n"

    unet_report_file = open(unet_report_path, "w")
    unet_report_file.write(unet_report_str)
    unet_report_file.close()

if __name__ == "__main__":
    main()