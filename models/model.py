import torch
from torch import nn

from backbone import Backbone
from RPN import RPN
from head import FastRCNNHead
from utils.generate_proposals import generate_proposals
from utils.generate_anchors import generate_anchors
from utils.roi_align import roi_align


class FasterRCNN(nn.Module):
    def __init__(self, img_size, num_classes, output_size=(7, 7)):
        super().__init__()
        self.img_size = img_size  # tuple
        self.backbone = Backbone()
        self.output_size = output_size
        self.rpn = RPN(self.backbone.out_channels)
        self.head = FastRCNNHead(self.backbone.out_channels, num_classes, output_size)

    def forward(self, images):
        """
        images: tensor, shape [B, 3, H, W], original image
        return:
            cls_score: List[Tensor[N_i, num_classes + 1]]
            bbox_pred: List[Tensor[N_i, num_classes * 4]]
            proposals: List[Tensor[N_i, 4]], N_i represents the number of remaining anchors for the i-th sample in the batch.
            cls_logits: tensor, shape [B, 2 * k, H/stride_H, W/stride_W], The output of the RPN network
            bbox_pred: tensor, shape [B, 4 * k, H/stride_H, W/stride_W], The output of the RPN network
            anchors: tensor, shape [num_anchors, 4]
        """
        feature_map = self.backbone(images)
        H_f, W_f = feature_map.shape[2], feature_map.shape[3]
        stride_h, stride_w = self.img_size[0] / H_f, self.img_size[1] / W_f
        anchors = generate_anchors((H_f, W_f), (stride_h, stride_w), img_size=self.img_size, device=feature_map.device)
        rpn_cls, rpn_bbox = self.rpn(feature_map)
        proposals = generate_proposals(rpn_cls, rpn_bbox, anchors, self.img_size)
        rois = roi_align(feature_map, proposals, self.img_size, output_size=self.output_size)
        cls_score, bbox_pred = [], []
        for i in range(len(rois)):
            one_cls_score, one_bbox_pred = self.head(rois[i])
            cls_score.append(one_cls_score), bbox_pred.append(one_bbox_pred)
        return cls_score, bbox_pred, proposals, rpn_cls, rpn_bbox, anchors

if __name__ == '__main__':
    images = torch.randn(4, 3, 256, 256)
    model = FasterRCNN(img_size=(256, 256), num_classes=6)
    cls_score, bbox_pred, proposals, rpn_cls, rpn_bbox, anchors = model(images)
    for one_cls_score, one_bbox_pred in zip(cls_score, bbox_pred):
        print(f"one_cls_score shape: {one_cls_score.shape}, one_bbox_pred shape: {one_bbox_pred.shape}")
    for proposal in proposals:
        print(f"proposal shape: {proposal.shape}")
    print(f"rpn_cls shape: {rpn_cls.shape}, rpn_bbox shape: {rpn_bbox.shape}")
    print(f"anchors shape: {anchors.shape}")


