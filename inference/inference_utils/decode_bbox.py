import torch

def decode_bbox(rois, deltas):
    """
    rois: tensor, shape [N_i, 4]
    deltas: tensor, shape [N_i, 4], (tx, ty, tw, th)
    return:
        tensor, shape [N_i, 4], (x1, y1, x2, y2)
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


if __name__ == '__main__':
    rois = torch.rand(10, 4)
    deltas = torch.rand(10, 4)
    boxes = decode_bbox(rois, deltas)
    print(boxes.shape)