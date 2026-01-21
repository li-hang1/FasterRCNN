import torch

from models.model import FasterRCNN
from utils import show_image_with_boxes


import torch
import torch.nn.functional as F
from torchvision.ops import nms
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

def decode_bbox(rois, deltas):
    """
    rois: [N, 4]
    deltas: [N, 4] (tx, ty, tw, th)
    """
    widths  = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    ctr_x = rois[:, 0] + 0.5 * widths
    ctr_y = rois[:, 1] + 0.5 * heights

    dx, dy, dw, dh = deltas.unbind(dim=1)

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = widths * torch.exp(dw)
    pred_h = heights * torch.exp(dh)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    return torch.stack([x1, y1, x2, y2], dim=1)












num_classes=6
model = FasterRCNN(img_size=(640, 640), num_classes=num_classes)
model.load_state_dict(torch.load("pretrained/faster_r_cnn.pth"))
model.eval()

cls_score, bbox_pred, proposals, rpn_cls, rpn_bbox, anchors = model()