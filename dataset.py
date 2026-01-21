import torch
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    """
    The dataset is a .pt file with the format {"images": images, "targets": targets}.
    images: tensor, shape [B, C, H, W], B represents the total number of images in the dataset.
    targets: List[{"boxes": boxes, "labels": labels}, ...], The list length is equal to B.
    boxes: tensor, shape [N_i, 4], (x1, y1, x2, y2), N_i represents the number of bounding boxes in each image.
    labels: tensor, shape (N_i, ), the class label for each bounding box.
    """
    def __init__(self, path):
        self.images = torch.load(path)["images"]
        self.targets = torch.load(path)["targets"]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(targets)

