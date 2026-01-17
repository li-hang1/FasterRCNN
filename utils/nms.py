import torch


def nms(boxes, scores, iou_threshold=0.7):
    """
    boxes: tensor, shape [num_anchors, 4], (x1, y1, x2, y2)
    scores: tensor, shape [num_anchors]
    iou_threshold: float
    reutrn: tensor, shape [num_rest_anchors], The indices of the remaining boxes in the input boxes.
    """
    _, indices = torch.sort(scores, descending=True)
    keep = []
    while indices.numel() > 0:
        current = indices[0].item()
        keep.append(current)
        if indices.numel() == 1:
            break
        rest = indices[1:]
        x1 = torch.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = torch.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = torch.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = torch.minimum(boxes[current, 3], boxes[rest, 3])
        inter_w, inter_h = (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)
        inter_area = inter_w * inter_h
        area_current = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter_area / (area_current + area_rest - inter_area)
        indices = rest[iou <= iou_threshold]
    return torch.tensor(keep, device=boxes.device)


if __name__ == '__main__':
    # 完全不重叠
    boxes = torch.tensor([
        [0, 0, 10, 10],
        [20, 20, 30, 30],
        [40, 40, 50, 50],
    ], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8, 0.7])
    print(nms(boxes, scores, iou_threshold=0.5))
    # 完全重叠
    boxes = torch.tensor([
        [0, 0, 10, 10],
        [0, 0, 10, 10],
        [0, 0, 10, 10],
    ], dtype=torch.float)
    scores = torch.tensor([0.3, 0.9, 0.6])
    print(nms(boxes, scores, iou_threshold=0.5))
    # 部分重叠
    boxes = torch.tensor([
        [0, 0, 10, 10],
        [1, 1, 11, 11],
        [20, 20, 30, 30],
    ], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8, 0.7])
    print(nms(boxes, scores, iou_threshold=0.5))



