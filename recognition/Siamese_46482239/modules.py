import torch
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_dim, dim, kernel_size, stride=1, dropout_p: float = 0.2):
        super().__init__()
        self.features = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, stride=stride, bias=True)
        self.norm1 = nn.GroupNorm(min(8, dim), dim)
        self.act_layer = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        out = self.features(x)
        out = self.norm1(out)
        out = self.act_layer(out)
        out = self.dropout(out)
        return out

def make_stages(in_channels: int, dim: int, n_stages: int, dropout_p: float):
    """
    n_stages stride-2 conv blocks; channels double each stage (dim, 2*dim, 4*dim, ...).
    At 224x224 input (3x3 convs, no padding) the spatial size goes 224->111->55->27->13->6 for n_stages=5,
    and the receptive field of the last layer grows to 63 px (vs 11 px for your 4-block stride-1/1/2/2 stack).
    """
    chans = [in_channels] + [dim * 2 ** i for i in range(n_stages)]
    blocks = [BasicBlock(chans[i], chans[i + 1], 3, stride=2, dropout_p=dropout_p) for i in range(n_stages)]
    return nn.Sequential(*blocks), chans[-1]


class CNN(nn.Module):
    """Baseline classifier. Shares the SAME backbone as CNNEmbed so the comparison is fair."""
    def __init__(self, dim, in_channels, num_class=2, dropout_p=0.2, n_stages=5):
        super().__init__()
        self.stages, self.pooling_depth = make_stages(in_channels, dim, n_stages, dropout_p)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.pooling_depth, num_class)

    def forward(self, x):
        out = self.avg_pool(self.stages(x))
        return self.classifier(out.view(out.size(0), -1))


class CNNEmbed(nn.Module):
    def __init__(self, dim, in_channels: int, embed_dims: int = 128, dropout_p: float = 0.2,
                 normalise: bool = True, n_stages: int = 5):
        super().__init__()
        self.normalise = normalise
        self.stages, self.pooling_depth = make_stages(in_channels, dim, n_stages, dropout_p)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(self.pooling_depth, embed_dims)

    def forward(self, x) -> torch.Tensor:
        out = self.avg_pool(self.stages(x))
        out = self.embed(out.view(out.size(0), -1))
        if self.normalise:
            out = F.normalize(out, dim=1)   # unit-length embeddings
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