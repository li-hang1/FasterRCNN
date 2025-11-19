import torch

def clip_anchors_to_image(anchors, image_size):
    H, W = image_size
    x1 = anchors[:, 0].clamp(min=0, max=W)
    y1 = anchors[:, 1].clamp(min=0, max=H)
    x2 = anchors[:, 2].clamp(min=0, max=W)
    y2 = anchors[:, 3].clamp(min=0, max=H)
    return torch.stack([x1, y1, x2, y2], dim=1)