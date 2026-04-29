import os
import argparse
import time
import yaml

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights


def parse_args():
    p = argparse.ArgumentParser(description="Train Faster R-CNN for constellation detection")
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


class ConstellationDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes = []
        labels = []
        areas = []

        for ann in coco_annotation:
            xmin = ann["bbox"][0]
            ymin = ann["bbox"][1]
            xmax = xmin + ann["bbox"][2]
            ymax = ymin + ann["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann["category_id"])
            areas.append(ann.get("area", 0.0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        img_id = torch.tensor([img_id])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes, pretrained_base="fasterrcnn_resnet50_fpn"):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("train_rcnn", {})
    rcnn_cfg = cfg.get("rcnn", {})

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_root = train_cfg.get("data_root", "data/constellation_dataset")
    train_json = train_cfg.get("train_json")
    val_json = train_cfg.get("val_json")
    train_img_dir = train_cfg.get("train_img_dir")
    val_img_dir = train_cfg.get("val_img_dir")

    epochs = int(train_cfg.get("epochs", 100))
    batch_size = int(train_cfg.get("batch_size", 1))
    lr = float(train_cfg.get("lr", 5e-3))
    weight_decay = float(train_cfg.get("weight_decay", 5e-4))
    num_workers = int(train_cfg.get("num_workers", 0))

    # dataset paths
    train_root = train_img_dir if train_img_dir else os.path.join(data_root, "images/train")
    val_root = val_img_dir if val_img_dir else os.path.join(data_root, "images/val")
    train_ann = train_json if train_json else os.path.join(data_root, "annotations/train.json")
    val_ann = val_json if val_json else os.path.join(data_root, "annotations/val.json")

    transforms = T.Compose([T.ToTensor()])

    train_dataset = ConstellationDataset(train_root, train_ann, transforms)
    val_dataset = ConstellationDataset(val_root, val_ann, transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)

    # num_classes in config is number of categories (no background). FasterRCNN expects num_classes including background.
    cfg_num = int(rcnn_cfg.get("num_classes", 17))
    num_classes = cfg_num + 1

    model = build_model(num_classes, pretrained_base=rcnn_cfg.get("pretrained_base"))
    model.to(device)

    # Freeze backbone if requested
    for param in model.backbone.body.parameters():
        param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(os.path.join("models", "rcnn"), exist_ok=True)

    start_epoch = 0
    interrupted = False

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0
            batches = 0

            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += losses.item()
                batches += 1

            avg_loss = epoch_loss / max(batches, 1)
            lr_scheduler.step()
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

            # save checkpoint
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            ckpt_path = os.path.join("models", "rcnn", f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            # also save scripted .pt for quick inference on early epochs
            if (epoch + 1) <= 15:
                model.eval()
                scripted = torch.jit.script(model)
                pt_path = os.path.join("models", "rcnn", "constellation_rcnn.pt")
                scripted.save(pt_path)
                print(f"Saved scripted model: {pt_path}")
                model.train()

    except KeyboardInterrupt:
        print("Training interrupted by user — saving resume checkpoint...")
        interrupted = True

    # final save
    final_path = os.path.join("models", "rcnn", "constellation_rcnn.pth")
    torch.save({"model": model.state_dict()}, final_path)
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
