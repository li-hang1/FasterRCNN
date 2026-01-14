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
        """
        feature_map: The output of backbone. [B, backbone_out_channels, H/stride_H, W/stride_W]
                    stride_H and stride_W represent the scaling ratio of the backbone output relative to the original image, respectively.
        return:
            cls_logits: [B, 2 * k, H/stride_H, W/stride_W]
            bbox_pred: [B, 4 * k, H/stride_H, W/stride_W]
        """
        x = F.relu(self.conv(feature_map))
        cls_logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logits, bbox_pred


if __name__ == '__main__':
    import torch
    x = torch.randn(4, 2048, 16, 16)
    model = RPN(in_channels=2048)
    cls_logits, bbox_pred = model(x)
    print(f"cls_logits shape: {cls_logits.shape}, bbox_pred: {bbox_pred.shape}")