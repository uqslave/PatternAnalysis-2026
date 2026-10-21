import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_dim, dim, kernel_size, stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, dropout_p=0.3):
        super().__init__()
        self.features = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, stride=stride, bias=True)
        self.norm1 = norm_layer(dim)
        self.act_layer = act_layer()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        out = self.norm1(self.features(x))
        out = self.act_layer(out)
        out = self.dropout(out)
        return out

class ConvNetwork(nn.Module):
    def __init__(self, dim, in_channels, num_classes=10, dropout_p=0.2):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pooling_depth = 2*self.dim

        self.block1 = BasicBlock(self.in_channels, self.dim, 3, stride=2, dropout_p=dropout_p)
        self.block2 = BasicBlock(self.dim, self.dim*2, 3, stride=2, dropout_p=dropout_p)

        self.ave_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(self.pooling_depth, self.num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        out = self.ave_pool(x)
        out = out.view(out.size(0), -1) #flatten

        return self.classifier(out)

class SiameseNetwork(nn.Module):
    def __init__(self, dim, num_classes=10, dropout_p=0.2):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.cnn = ConvNetwork(dim, self.in_channels, self.num_classes, dropout_p)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.dim, self.num_classes)

    def forward(self, x, y):
        x = self.cnn(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

