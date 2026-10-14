import torch
from torch import nn

class UNet(nn.Module):
    """
    UNet model class.
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super().__init__()

        # Encoder (downsampling)
        self.encoder1 = self.conv_block(in_channels, 32, dropout_p=dropout_p)
        self.encoder2 = self.conv_block(32, 64, dropout_p=dropout_p)
        self.encoder3 = self.conv_block(64, 128, dropout_p=dropout_p)
        self.encoder4 = self.conv_block(128, 256, dropout_p=dropout_p)

        # Decoder (upsampling)
        self.decoder4 = self.conv_block(256 + 128, 128, dropout_p=dropout_p)
        self.decoder3 = self.conv_block(128 + 64, 64, dropout_p=dropout_p)
        self.decoder2 = self.conv_block(64 + 32, 32, dropout_p=dropout_p)
        self.decoder1 = nn.Conv2d(32, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Softmax2d()#nn.Sigmoid()     # Sigmoid act func for final output

    def conv_block(self, in_channels, out_channels, dropout_p=0.2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout_p),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Decoder
        d4 = self.decoder4(torch.cat([self.upsample(e4), e3], dim=1))
        d3 = self.decoder3(torch.cat([self.upsample(d4), e2], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e1], dim=1))
        out = self.decoder1(d2)

        # Apply sigmoid activation to final output
        out = self.sigmoid(out)

        return out

import torch.nn.functional as F
class DiceLoss(nn.Module):
    """
    Dice Loss (DSC) class.
    """
    def __init__(self, smooth: float= 1.):
        """
        :param smooth: Prevent division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        :param predictions: Sigmoid output from model [B, C, H, W] (values in [0, 1])
        :param targets: Ground truth [B, C, H, W] (values in {0, 1})
        :return: Dice loss (DSC)
        """
        predictions_shape = predictions.shape
        targets_shape = targets.shape
        assert predictions_shape == targets_shape, "Predictions and Targets must have the same shape!"
        assert predictions_shape[1] == targets_shape[1], "Channel mismatch between Predictions and Targets"

        # Channels (K)
        # 0 -> Background
        # 1 -> Body outline
        # 2 -> Bone
        # 3 -> Bladder
        # 4 -> Rectum
        # 5 -> Prostate

        # Flattening the prediction and target tensors into single array for each batch and class/channel
        # Spatial input irrelevant since DSC is counting how many pixels overlap with correct result (ground truth)
        B, K, _, _ = predictions_shape
        predictions = predictions.reshape(B, K, -1)
        targets = targets.reshape(B, K, -1).float()

        # Calculate intersection and union
        intersection = (predictions * targets).sum(dim=2)                           # [B, K]
        union = predictions.sum(dim=2) + targets.sum(dim=2)                         # [B, K]
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)   # [B, K]

        return 1 - dice_per_class.mean()


@torch.no_grad()
def dsc_per_class(predictions, targets, smooth=1):
    """
    Returns a tensor of shape [K] with the Dice score for each class, averaged over the batch.
    """
    B, C, _, _ = predictions.shape
    #predictions = (predictions > 0.5).float()
    predictions = predictions.reshape(B, C, -1)
    targets = targets.reshape(B, C, -1).float()

    intersection = (predictions * targets).sum(dim=2)
    union = predictions.sum(dim=2) + targets.sum(dim=2)
    dice = (2 * intersection + smooth) / (union + smooth)   # [B, K]

    return dice.mean(dim=0)                                 # [K] -- one score per class
