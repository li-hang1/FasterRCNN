def show_image_with_boxes(img, target):
    if torch.is_tensor(img):
        img = img.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(img)
    boxes = target["boxes"]
    labels = target["labels"]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-2, f"cls {labels[i].item()}", color="yellow", fontsize=12, backgroundcolor="black")
    plt.show()

class DetectionDataset(Dataset):
    def __init__(self, path):
        self.images = torch.load(path)["images"]
        self.targrts = torch.load(path)["targets"]
    def __len__(self):
        return self.images[0]
    def __getitem__(self, idx):
        return self.images[idx], self.targrts[idx]

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(targets)

dataset = DetectionDataset("/root/LH/object_detection.pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 500
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
            label, bbox_target = assign_proposals_to_gt(proposal, target["boxes"], target["labels"])
            labels.append(label)
            bbox_targets.append(bbox_target)
        loss_rpn = rpn_loss(rpn_cls, rpn_bbox, anchors, [target["boxes"] for target in targets])
        loss_rcnn = fast_rcnn_loss(cls_score, bbox_pred, labels, bbox_targets, num_classes=num_classes)
        loss = loss_rpn + loss_rcnn
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
        model_path = "/root/LH/faster_r_cnn.pth"
        torch.save(model.state_dict(), model_path)