import torch
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_dim, dim, kernel_size, stride=1, dropout_p: float = 0.2):
        super().__init__()
        self.features = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, stride=stride, bias=True)
        self.norm1 = nn.GroupNorm(dim, dim)
        self.act_layer = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        out = self.features(x)
        out = self.norm1(out)
        out = self.act_layer(out)
        out = self.dropout(out)
        return out

class CNN(nn.Module):
    def __init__(self, dim, in_channels, num_class=2, dropout_p=0.2):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.num_class = num_class
        self.pooling_depth = 2 * self.dim

        self.block1 = BasicBlock(self.in_channels, self.dim, 3, stride=2, dropout_p=dropout_p)
        self.block2 = BasicBlock(self.dim, self.dim * 2, 3, stride=2, dropout_p=dropout_p)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.pooling_depth, self.num_class)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)  # flatten
        return self.classifier(out)

class CNNEmbed(nn.Module):
    def __init__(self, dim, in_channels: int, embed_dims: int = 128, dropout_p: float = 0.2, normalise: bool = True):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.normalise = normalise
        self.pooling_depth = 2 * self.dim

        self.block1 = BasicBlock(self.in_channels, self.dim, 3, stride=2, dropout_p=dropout_p)
        self.block2 = BasicBlock(self.dim, self.dim * 2, 3, stride=2, dropout_p=dropout_p)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(self.pooling_depth, self.embed_dims)

    def forward(self, x) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)  # flatten
        out = self.embed(out)
        if self.normalise:
            out = F.normalize(out, dim = 1) # Optional: normalises embedding
        return out


class SiameseNetwork(nn.Module):
    def __init__(self, encoder: CNNEmbed):
        super().__init__()
        self.encoder = encoder

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out1 = self.encoder(x1)
        out2 = self.encoder(x2)
        return out1, out2

class ContrastiveLoss(nn.Module):
    """
    Hadsell et al. (2006). Label convention (matches your original formula):
        pair = 0 -> SAME class      -> loss = D^2              (pull together)
        pair = 1 -> DIFFERENT class -> loss = max(0, m - D)^2  (push apart until margin m)
    """
    def __init__(self, margin = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, x: torch.Tensor, y: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        D = F.pairwise_distance(x, y, eps=1e-8)
        pair = pair.float()
        loss = (1 - pair) * D.pow(2) + pair * torch.clamp(self.margin - D, min=0).pow(2)
        return loss.mean()  # scalar, so .backward() works