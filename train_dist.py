import torch
from torch.utils.data import DataLoader
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import DetectionDataset, collate_fn
from models.model import FasterRCNN
from utils.assign_proposals_to_gt import assign_proposals_to_gt
from losses.rpn_loss import rpn_loss
from losses.fast_rcnn_loss import fast_rcnn_loss


rank = int(os.environ.get('RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '0'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))

dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.cuda.set_device(local_rank)

is_main = (rank == 0)

dataset = DetectionDataset("data/dataset.pt")
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=20, pin_memory=True, collate_fn=collate_fn)

epochs = 100
num_classes=6
model = FasterRCNN(img_size=(640, 640), num_classes=num_classes).cuda()
model = DDP(model, device_ids=[local_rank])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if is_main:
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
    print(f"Total params: {total_params}")

ckpt_path = os.path.join("pretrained", "faster_r_cnn.pth")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt)
    if is_main:
        print(f"Load model from {ckpt_path}")

for epoch in range(1, epochs):
    sampler.set_epoch(epoch)
    model.train()
    for images, targets in dataloader:
        images = images.cuda()
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
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
    if is_main:
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
        model_path = "pretrained/faster_r_cnn.pth"
        torch.save(model.module.state_dict(), model_path)

dist.destroy_process_group()


