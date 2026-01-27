from torch import nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_channels, scale_ratio=(3, 3)):
        super().__init__()
        k = scale_ratio[0] * scale_ratio[1]
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.cls_logits = nn.Conv2d(512, k * 2, 1)
        self.bbox_pred = nn.Conv2d(512, k * 4, 1)

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