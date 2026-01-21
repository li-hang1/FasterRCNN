import torch
from torch.utils.data import DataLoader

from dataset import DetectionDataset, collate_fn
from models.model import FasterRCNN
from utils.assign_proposals_to_gt import assign_proposals_to_gt
from losses.rpn_loss import rpn_loss
from losses.fast_rcnn_loss import fast_rcnn_loss


dataset = DetectionDataset("data/dataset.pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
num_classes=6
model = FasterRCNN(img_size=(640, 640), num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(epochs):
    model.train()
    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        cls_score, bbox_pred, proposals, rpn_cls, rpn_bbox, anchors = model(images)
        labels, bbox_targets = [], []
        for proposal, target in zip(proposals, targets):
            label, bbox_target = assign_proposals_to_gt(proposal, target["boxes"], target["labels"]+1)
            labels.append(label)
            bbox_targets.append(bbox_target)
        loss_rpn = rpn_loss(rpn_cls, rpn_bbox, anchors, [target["boxes"] for target in targets])
        loss_rcnn = fast_rcnn_loss(cls_score, bbox_pred, labels, bbox_targets, num_classes=num_classes)
        loss = loss_rpn + loss_rcnn
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    model_path = "pretrained/faster_r_cnn.pth"
    torch.save(model.state_dict(), model_path)


