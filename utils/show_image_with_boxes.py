from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches


def show_image_with_boxes(img_path, dets, class_names=None):
    """
    img_path: path to image file
    dets: detection result, tensor, shape: [N_i, 6], (x1, y1, x2, y2, score, cls)
    class_names: list of class names
    """
    img = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(img)
    if dets.numel() > 0:
        for *box, score, cls in dets.cpu().numpy():
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            if class_names is not None:
                label = f"{class_names[int(cls)-1]}: {score:.2f}"
            else:
                label = f"{int(cls)}: {score:.2f}"
            ax.text(x1, y1-5, label, color="yellow", fontsize=8)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    import os
    import torch
    img_dir = "../data/images"
    label_dir = "../data/labels"
    class_names = ["apple", "banana", "grape", "orange", "pineapple", "watermelon"]
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        dets = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, w, h = line.strip().split()
                cls, x, y, w, h = map(float, [cls, x, y, w, h])
                x, y, w, h = x*640, y*640, w*640, h*640
                x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
                dets.append([x1, y1, x2, y2, 1.0, cls])
        dets = torch.tensor(dets)
        show_image_with_boxes(img_path, dets, class_names)

