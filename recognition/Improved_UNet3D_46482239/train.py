import os
import numpy as np
import torch
import random
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from dataset import LABELS, get_dataloader_2d
from modules import UNet, dsc_per_class, DiceCELoss, TverskyCELoss, ImprovedUnet

# Set random seeds for reproducibility
torch.manual_seed(67)
np.random.seed(67)
random.seed(67)
torch.cuda.manual_seed(67)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device.type)

train_dataloader = get_dataloader_2d("train")
validate_dataloader = get_dataloader_2d("validate")

in_channels = 1 # Greyscale
out_channels = len(LABELS)
dropout = 0.23
model = ImprovedUnet(in_channels, out_channels, dropout)
model = model.to(device)

num_epochs = 40
learning_rate = 0.001
optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_epochs, eta_min=0.00007)
criterion = DiceCELoss(w=0.5)#DiceCELoss(w=0.5) #TverskyCELoss(a=0.26, b=0.74, w=0.5) #


output_dir = Path("audit_out/")
output_dir.mkdir(parents=True, exist_ok=True)

losses = []
val_losses = []
accuracy_scores = []
val_accuracy_scores = []
unet_report_train_temp = str()
unet_report_validation_temp = str()
unet_report_test_temp = str()
unet_report_path = output_dir / r"unet_report.txt"

print("Training...")
start_time = time.time()
training_times = [0.0]
training_subtime = start_time
for epoch in range(num_epochs):
    # ======== Training Segment ========
    model.train()
    epoch_loss = validation_loss = 0
    losses_per_batch = torch.Tensor().to(device)
    losses_per_batch_temp = torch.Tensor().to(device)

    for batch_idx, (images, masks) in enumerate(train_dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimiser.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimiser.step()

        losses_per_batch = torch.cat([losses_per_batch, dsc_per_class(outputs, masks).reshape(-1, len(LABELS))])
        epoch_loss = epoch_loss + loss.item()

    unet_report_train_temp = unet_report_train_temp + f"Mean DSC per class at epoch {epoch + 1}/{num_epochs}\n"
    losses_per_batch = torch.mean(losses_per_batch, dim=0)
    for label_idx, label in enumerate(LABELS):
        unet_report_train_temp = unet_report_train_temp + f"{label}: {losses_per_batch[label_idx]:.3f}\n"
    unet_report_train_temp = unet_report_train_temp + f"\n"

    avg_loss = epoch_loss / len(train_dataloader)
    losses.append(avg_loss)
    accuracy_scores.append(1 - avg_loss)
    out_str = f"Epoch {epoch + 1}/{num_epochs} | Average Loss: {avg_loss:.3f}"

    training_subtime = time.time() - training_subtime
    training_times.append(training_subtime)

    # ======== Validate Segment ========
    model.eval()
    with torch.no_grad():
        for val_images, val_masks in validate_dataloader:
            val_images = val_images.to(device)
            val_masks = val_masks.to(device)

            outputs = model(val_images)
            val_loss = criterion(outputs, val_masks)
            validation_loss = validation_loss + val_loss.item()
            losses_per_batch_temp = torch.cat(
                [losses_per_batch_temp, dsc_per_class(outputs, val_masks).reshape(-1, len(LABELS))]
            )

        unet_report_validation_temp = unet_report_validation_temp + f"Mean DSC per class for validation set at epoch {epoch + 1}/{num_epochs}\n"
        losses_per_batch_temp = torch.mean(losses_per_batch_temp, dim=0)
        for label_idx, label in enumerate(LABELS):
            unet_report_validation_temp = unet_report_validation_temp + f"{label}: {losses_per_batch_temp[label_idx]:.3f}\n"
        unet_report_validation_temp = unet_report_validation_temp + f"\n"
    scheduler.step()
    avg_validation_loss = validation_loss / len(validate_dataloader)
    val_losses.append(avg_validation_loss)
    val_accuracy_scores.append(1 - avg_validation_loss)
    out_str = out_str + f" | Validation Loss: {avg_validation_loss:.3f}\tTime: {training_subtime:.3f}s"
    training_subtime = time.time() - training_subtime
    print(out_str)


outpath = output_dir / f"unet_improved_dsc.pth"
torch.save(model.state_dict(), outpath)
print(f"Saved checkpoint to {outpath}")
print("")

# Plot training loss vs validation loss
fig, ax1 = plt.subplots(figsize=(700, 700, "px"))
ax1.plot(losses, color='tab:red', label='Training Loss')
ax1.plot(val_losses, color='tab:orange', label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.plot(accuracy_scores, color='tab:blue', label='Training DSC Score')
ax2.plot(val_accuracy_scores, color='tab:green', label='Validation DSC Score')
ax2.set_ylabel('DSC Score', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
fig.legend(loc="lower left")

plt.title("UNET Segmentation Training Curve")
fig.tight_layout()
outpath = output_dir / r"UNET Segmentation Training Curve.png"
fig.savefig(outpath)
plt.show()

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

unet_report_str = unet_report_str + unet_report_test_temp
unet_report_str = unet_report_str + "=" * divider_length + "\n\n"

unet_report_file = open(unet_report_path, "w")
unet_report_file.write(unet_report_str)
unet_report_file.close()