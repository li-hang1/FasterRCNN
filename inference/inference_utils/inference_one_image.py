from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F

from utils.nms import nms
from utils.show_image_with_boxes import show_image_with_boxes
from .decode_bbox import decode_bbox


@torch.no_grad()
def inference_one_image(model, img_path, class_names, num_classes=6, score_thresh=0.5, nms_thresh=0.5, device=torch.device("cpu")):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)

    cls_scores, bbox_preds, proposals, _, _, _ = model(img_tensor)

    cls_score = cls_scores[0]        # [N, num_classes+1]
    bbox_pred = bbox_preds[0]        # [N, num_classes*4]
    rois = proposals[0]              # [N, 4]

    probs = F.softmax(cls_score, dim=1)  # [N, num_classes+1]

    results = []
    for cls in range(1, num_classes + 1):
        cls_prob = probs[:, cls]
        keep = cls_prob > score_thresh
        if keep.sum() == 0:
            continue

        scores = cls_prob[keep]
        rois_keep = rois[keep]

        deltas = bbox_pred[keep, (cls-1)*4:cls*4]
        boxes = decode_bbox(rois_keep, deltas)

        boxes[:, 0::2].clamp(min=0, max=w)
        boxes[:, 1::2].clamp(min=0, max=h)

        keep_idx = nms(boxes, scores, nms_thresh)

        for i in keep_idx:
            results.append(torch.stack([boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i], torch.tensor(cls, device=device)]))

    if len(results) == 0:
        dets = torch.zeros((0, 6), device=device)
        show_image_with_boxes(img_path, dets, class_names)
    else:
        dets = torch.stack(results)
        show_image_with_boxes(img_path, dets, class_names)
