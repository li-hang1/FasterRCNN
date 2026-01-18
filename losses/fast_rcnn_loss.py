import torch
import torch.nn.functional as F

def fast_rcnn_loss(class_logits, bbox_pred, gt_labels, gt_boxes, num_classes, lambda_reg=1.0):
    all_cls_loss, all_reg_loss = []