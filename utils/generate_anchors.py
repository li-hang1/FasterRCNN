import torch
import numpy as np
from clip_anchors_to_image import clip_anchors_to_image

def generate_anchors(feature_map_size, strides, img_size, scales=(64, 128, 256), ratios=(0.5, 1, 2), device='cpu'):
    """
    feature_map_size: tuple (backbone_H, backbone_W), The height and width of the output of Backbone.
    img_size: tuple (H, W), The height and width of the original image.
    strides: tuple (stride_H, stride_W), stride_H and stride_W represent the scaling ratio of the backbone output relative to the original image, respectively.
    return:
        tensor [num_anchors, 4], The coordinates of the top left and bottom right corners of the anchor.
    """
    backbone_H, backbone_W = feature_map_size
    stride_H, stride_W = strides
    anchors = []
    for i in range(backbone_H):
        for j in range(backbone_W):
            ctr_x = j * stride_W + stride_W / 2
            ctr_y = i * stride_H + stride_H / 2
            for scale in scales:
                for ratio in ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    x1 = ctr_x - w / 2
                    y1 = ctr_y - h / 2
                    x2 = ctr_x + w / 2
                    y2 = ctr_y + h / 2
                    anchors.append([x1, y1, x2, y2])
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = clip_anchors_to_image(anchors, img_size)
    return anchors

if __name__ == '__main__':
    anchors = generate_anchors(feature_map_size=(16, 16), strides=(32, 32), img_size=(512, 512))
    print(anchors.shape)

