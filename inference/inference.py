import torch
import os

from models.model import FasterRCNN
from inference_utils.inference_one_image import inference_one_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 6
model = FasterRCNN(img_size=(640, 640), num_classes=num_classes).to(device)
model.load_state_dict(torch.load("../pretrained/faster_r_cnn.pth", map_location=device))

class_names = ["apple", "banana", "grape", "orange", "pineapple", "watermelon"]
for demo_img in os.listdir("../demos"):
    img_path = os.path.join("../demos", demo_img)
    inference_one_image(model, img_path=img_path, class_names=class_names, num_classes=num_classes, score_thresh=0.9, nms_thresh=0.1, device=device)



