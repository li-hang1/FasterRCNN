import os
import torch
from PIL import Image
import torchvision.transforms as transforms

img_dir = "../data/images"
label_dir = "../data/labels"
save_dir = "data/dataset.pt"

transform = transforms.Compose([transforms.ToTensor()])
images, targets = [], []

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    images.append(image)

    boxes, labels = [], []
    with open(label_path, "r") as f:
        for line in f:
            cls, x, y, w, h = line.strip().split()
            cls = int(cls)
            x, y, w, h = map(float, [x, y, w, h])
            labels.append(int(cls))
            boxes.append([x-w/2, y-h/2, x+w/2, y+h/2])
    boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
    target = {"boxes": boxes, "labels": labels}
    targets.append(target)
images = torch.stack(images, dim=0)
torch.save({"images": images, "targets": targets}, save_dir)
print(f"Saved Complete {images.shape}, {len(targets)}")







