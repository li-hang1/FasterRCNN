import torch

def bbox2delta(boxes, gt_boxes):
    """
    boxes shape: [N, 4], (x1, y1, x2, y2)
    gt_boxes shape: [N, 4], (x1, y1, x2, y2)
    return:
        shape: [N, 4], (dx, dy, dw, dh)
    """
    px = (boxes[:, 0] + boxes[:, 2]) * 0.5
    py = (boxes[:, 1] + boxes[:, 3]) * 0.5
    pw = boxes[:, 2] - boxes[:, 0]
    ph = boxes[:, 3] - boxes[:, 1]

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = torch.log(gw / pw)
    th = torch.log(gh / ph)

    deltas = torch.stack([tx, ty, tw, th], dim=1)
    return deltas

if __name__ == "__main__":
    boxes = torch.randn(100, 4)
    gt_boxes = torch.randn(100, 4)
    deltas = bbox2delta(boxes, gt_boxes)
    print(deltas.shape)