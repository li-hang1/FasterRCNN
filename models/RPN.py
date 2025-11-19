from torch import nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_channels, anchor_scales=[64, 128, 256], anchor_ratios=[0.5, 1.0, 2.0]):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.k = len(anchor_scales) * len(anchor_ratios)
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.cls_logits = nn.Conv2d(512, self.k * 2, 1)
        self.bbox_pred = nn.Conv2d(512, self.k * 4, 1)
    def forward(self, feature_map):
        x = F.relu(self.conv(feature_map))
        cls_logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logits, bbox_pred
