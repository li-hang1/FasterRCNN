import torch

def apply_deltas_to_anchors(anchors, deltas):
    """
    anchors: tensor [num_anchors, 4], (x1, y1, x2, y2)
    deltas: tensor [num_anchors, 4], (dx, dy, dw, dh)
    return:
        modified anchors, tensor [num_anchors, 4]
    """
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = widths * torch.exp(dw)
    pred_h = heights * torch.exp(dh)

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes

if __name__ == "__main__":
    anchors = torch.randn(1000, 4)
    deltas = torch.randn(1000, 4)
    pred_boxes = apply_deltas_to_anchors(anchors, deltas)
    print(pred_boxes.shape)