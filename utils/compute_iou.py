import torch

def compute_iou(proposals, gt_boxes):
    """
    proposals shape: [num_proposals, 4], (x1, y1, x2, y2)
    gt_boxes shape: [num_gt_boxes, 4], (x1, y1, x2, y2)
    return:
        shape: [num_proposals, num_gt_boxes]
    """
    proposals, gt_boxes = proposals[:, None, :], gt_boxes[None, :, :]

    xx1 = torch.max(proposals[..., 0], gt_boxes[..., 0])           # [num_proposals, num_gt_boxes]
    yy1 = torch.max(proposals[..., 1], gt_boxes[..., 1])
    xx2 = torch.min(proposals[..., 2], gt_boxes[..., 2])
    yy2 = torch.min(proposals[..., 3], gt_boxes[..., 3])

    inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)    # [num_proposals, num_gt_boxes]

    area_boxes = (proposals[..., 2] - proposals[..., 0]) * (proposals[..., 3] - proposals[..., 1])
    area_gt = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])

    union = area_boxes + area_gt - inter                           # [num_proposals, num_gt_boxes]
    iou = inter / union.clamp(min=1e-6)

    return iou

if __name__ == "__main__":
    proposals = torch.rand(20, 4)
    gt_boxes = torch.rand(10, 4)
    iou = compute_iou(proposals, gt_boxes)
    print(iou.shape)