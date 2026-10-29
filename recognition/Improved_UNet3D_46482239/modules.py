import torch
from torch import nn


class ResidualConvBlock3D(nn.Module):
    """
    3D Residual convolution block.
    """

    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(dropout_p)
        )
        self.block2_1 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
        )
        self.block2_2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(dropout_p)
        )

        # Shortcut connection to match dimensions if in_channels != out_channels
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block1(x)
        out = self.block2_1(out)
        out = out + residual    # Add residual shortcut before final activation
        out = self.block2_2(out)
        return out


class ImprovedUnet3D(nn.Module):
    """
    Improved UNet 3D model class using residual blocks
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super().__init__()

        # Encoder (downsampling) using Residual Blocks
        self.encoder1 = ResidualConvBlock3D(in_channels, 32, dropout_p=dropout_p)
        self.encoder2 = ResidualConvBlock3D(32, 64, dropout_p=dropout_p)
        self.encoder3 = ResidualConvBlock3D(64, 128, dropout_p=dropout_p)
        self.encoder4 = ResidualConvBlock3D(128, 256, dropout_p=dropout_p)

        # Decoder (upsampling) using Residual Blocks
        self.decoder4 = ResidualConvBlock3D(256 + 128, 128, dropout_p=dropout_p)
        self.decoder3 = ResidualConvBlock3D(128 + 64, 64, dropout_p=dropout_p)
        self.decoder2 = ResidualConvBlock3D(64 + 32, 32, dropout_p=dropout_p)
        self.decoder1 = nn.Conv3d(32, out_channels, kernel_size=1)

        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.act = nn.Softmax(dim = 1)

        # Deep supervision: sum predictions from multiple scales
        self.seg4 = nn.Conv3d(128, out_channels, kernel_size=1)     # d4: 1/4 resolution
        self.seg3 = nn.Conv3d(64, out_channels, kernel_size=1)      # d3: 1/2 resolution

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Decoder with skip connections
        d4 = self.decoder4(torch.cat([self.upsample(e4), e3], dim=1))
        d3 = self.decoder3(torch.cat([self.upsample(d4), e2], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e1], dim=1))

        # Deep supervision
        s = self.seg4(d4)  # 1/4 res
        s = self.upsample(s) + self.seg3(d3)  # 1/2 res
        out = self.upsample(s) + self.decoder1(d2)  # full res

        # Apply activation to final output
        out = self.act(out)

        return out


class ResidualConvBlock(nn.Module):
    """
    2D Residual convolution block.
    """

    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout_p)
        )
        self.block2_1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
        )
        self.block2_2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout_p)
        )

        # Shortcut connection to match dimensions if in_channels != out_channels
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block1(x)
        out = self.block2_1(out)
        out = out + residual    # Add residual shortcut before final activation
        out = self.block2_2(out)
        return out


class ImprovedUnet(nn.Module):
    """
    Improved UNet model class using residual blocks
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super().__init__()

        # Encoder (downsampling) using Residual Blocks
        self.encoder1 = ResidualConvBlock(in_channels, 32, dropout_p=dropout_p)
        self.encoder2 = ResidualConvBlock(32, 64, dropout_p=dropout_p)
        self.encoder3 = ResidualConvBlock(64, 128, dropout_p=dropout_p)
        self.encoder4 = ResidualConvBlock(128, 256, dropout_p=dropout_p)

        # Decoder (upsampling) using Residual Blocks
        self.decoder4 = ResidualConvBlock(256 + 128, 128, dropout_p=dropout_p)
        self.decoder3 = ResidualConvBlock(128 + 64, 64, dropout_p=dropout_p)
        self.decoder2 = ResidualConvBlock(64 + 32, 32, dropout_p=dropout_p)
        self.decoder1 = nn.Conv2d(32, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.act = nn.Softmax2d()

        self.seg4 = nn.Conv2d(128, out_channels, kernel_size=1)     # d4: 1/4 resolution
        self.seg3 = nn.Conv2d(64, out_channels, kernel_size=1)      # d3: 1/2 resolution

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Decoder with skip connections
        d4 = self.decoder4(torch.cat([self.upsample(e4), e3], dim=1))
        d3 = self.decoder3(torch.cat([self.upsample(d4), e2], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e1], dim=1))

        # out = self.decoder1(d2)

        # Deep supervision
        s = self.seg4(d4)  # 1/4 res
        s = self.upsample(s) + self.seg3(d3)  # 1/2 res
        out = self.upsample(s) + self.decoder1(d2)  # full res

        # Apply activation to final output
        out = self.act(out)

        return out


class UNet(nn.Module):
    """
    Plain UNet model class.
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
        self.act = nn.Softmax2d()

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

        out = self.act(out)

        return out


class DiceCELoss(nn.Module):
    """
    Dice + cross-entropy.
    loss = w * Dice loss + (1 - w) * CE
    probs : softmax output [B, K, H, W]
    onehot: ground truth   [B, K, H, W]
    """
    def __init__(self, w: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.w = w
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, predictions, targets):
        targets = targets.float()
        ce = -(targets * torch.log(predictions.clamp(min=1e-3))).sum(dim=1).mean()   # per-pixel CE, averaged over B, H, W
        return self.w * self.dice(predictions, targets) + (1 - self.w) * ce


class DiceLoss(nn.Module):
    """
    Dice Loss (DSC) class.
    """
    def __init__(self, smooth: float= 0.5):
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
        B, K = predictions.shape[:2]
        predictions = predictions.reshape(B, K, -1) # Reshapes from [B, K, H, W] to [B, K, HxW]
        targets = targets.reshape(B, K, -1).float()

        # Calculate intersection and union
        intersection = (predictions * targets).sum(dim=2)                           # intersection.shape    [B, K]
        union = predictions.sum(dim=2) + targets.sum(dim=2)                         # union.shape           [B, K]
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)   # dsc.shape             [B, K]

        return 1 - dice_per_class.mean()


@torch.no_grad()
def dsc_per_class(predictions, targets, smooth=1e-8):
    """
    Used during inference.
    Returns a tensor of shape [K] with the Dice score for each class, averaged over the batch.
    :param predictions: Softmax output from model [B, C, H, W] (values in {0, 1})
    :param targets: Ground truth [B, C, H, W] (values in {0, 1})
    :param smooth: Prevent division by zero
    :return: Dice Score
    """
    B, K = predictions.shape[:2]
    # predictions = (predictions > 0.5).float() # Did argmax before calling this method.
    predictions = predictions.reshape(B, K, -1) # Becomes [B, K, HxW]
    targets = targets.reshape(B, K, -1).float()

    intersection = (predictions * targets).sum(dim=2)
    union = predictions.sum(dim=2) + targets.sum(dim=2)
    dice = (2 * intersection + smooth) / (union + smooth)   # [B, K]

    return dice.mean(dim=0)                                 # [K] -- one score per class
