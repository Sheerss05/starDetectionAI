import os
import sys
import signal
import time

# Add torch lib directory to DLL search path on Windows (required for CUDA DLLs)
if sys.platform == "win32":
    import importlib.util
    _torch_spec = importlib.util.find_spec("torch")
    if _torch_spec is not None:
        _torch_lib = os.path.join(os.path.dirname(_torch_spec.origin), "lib")
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)

import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

# =====================================
# 1. DEVICE SETUP
# =====================================

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("PyTorch Version:", torch.__version__)
print("Using device:", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print("GPU:", torch.cuda.get_device_name(0))

# =====================================
# 2. DATASET CLASS
# =====================================

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

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objs = len(coco_annotation)

        boxes = []
        labels = []
        areas = []

        for i in range(num_objs):

            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
            areas.append(coco_annotation[i]['area'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)

        img_id = torch.tensor([img_id])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        return img, target

    def __len__(self):
        return len(self.ids)

# =====================================
# 3. LOAD DATASET
# =====================================

dataset_path = "dataset"

train_root = os.path.join(dataset_path, "train")
train_ann = os.path.join(train_root, "_annotations.coco.json")

valid_root = os.path.join(dataset_path, "valid")
valid_ann = os.path.join(valid_root, "_annotations.coco.json")

transforms = T.Compose([T.ToTensor()])

train_dataset = ConstellationDataset(train_root, train_ann, transforms)
valid_dataset = ConstellationDataset(valid_root, valid_ann, transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print("Training images:", len(train_dataset))
print("Validation images:", len(valid_dataset))

# =====================================
# 4. MODEL SETUP
# =====================================

train_coco = COCO(train_ann)
num_classes = len(train_coco.getCatIds()) + 1

print("Number of classes:", num_classes)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

# Freeze backbone body (ResNet50) — skip its backprop, only train FPN + heads
for param in model.backbone.body.parameters():
    param.requires_grad = False

# =====================================
# 5. OPTIMIZER
# =====================================

params = [p for p in model.parameters() if p.requires_grad]

optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-4)

num_epochs = 50

lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)

# =====================================
# 5b. RESUME FROM CHECKPOINT
# =====================================

start_epoch = 0
_checkpoints = sorted(
    [f for f in os.listdir(".") if f.startswith("checkpoint_epoch_") and f.endswith(".pth")],
    key=lambda f: int(f.split("_")[-1].split(".")[0])
)
if _checkpoints:
    _latest = _checkpoints[-1]
    print(f"Found checkpoint: {_latest} — resuming...")
    _ckpt = torch.load(_latest, map_location=device)
    model.load_state_dict(_ckpt["model_state_dict"])
    optimizer.load_state_dict(_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in _ckpt:
        lr_scheduler.load_state_dict(_ckpt["scheduler_state_dict"])
    start_epoch = _ckpt["epoch"]
    print(f"Resumed from epoch {start_epoch}")
else:
    print("No checkpoint found — starting fresh")

# =====================================
# 6. TRAINING
# =====================================

_interrupted = False

def _handle_interrupt(sig, frame):
    global _interrupted
    print("\n\nCtrl+C detected — finishing batch then saving checkpoint...")
    _interrupted = True

signal.signal(signal.SIGINT, _handle_interrupt)

scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable params: {total_trainable:,}  |  Frozen (backbone): {total_frozen:,}")
print(f"Starting training from epoch {start_epoch + 1}/{num_epochs}...")

for epoch in range(start_epoch, num_epochs):

    if _interrupted:
        break

    model.train()
    epoch_loss = 0.0
    epoch_start = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="img", leave=True)

    for images, targets in pbar:

        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += losses.item()
        pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        if _interrupted:
            break

    lr_scheduler.step()

    elapsed   = time.time() - epoch_start
    avg_loss  = epoch_loss / len(train_loader)
    remaining = (num_epochs - epoch - 1) * elapsed
    print(
        f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} "
        f"| {elapsed/60:.1f} min/epoch | ETA: {remaining/3600:.1f} hr"
    )

    if True or _interrupted:  # save every epoch
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    if (epoch + 1) <= 15:
        model.eval()
        scripted = torch.jit.script(model)
        scripted.save("constellation_rcnn.pt")
        print(f"Model saved as constellation_rcnn.pt (epoch {epoch+1})")
        model.train()

    if _interrupted:
        print("Training paused. Run the script again to resume automatically.")
        sys.exit(0)

# =====================================
# 7. SAVE MODEL
# =====================================

torch.save(model.state_dict(), "constellation_rcnn.pth")
print("Model saved as constellation_rcnn.pth")

# Save as TorchScript .pt file
model.eval()
scripted = torch.jit.script(model)
scripted.save("constellation_rcnn.pt")
print("Model saved as constellation_rcnn.pt")

# =====================================
# 8. USER IMAGE UPLOAD
# =====================================

def detect_uploaded_image(model, device, threshold=0.5):

    model.eval()

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select a sky image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        print("No image selected.")
        return

    img = Image.open(file_path).convert("RGB")

    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    fig, ax = plt.subplots(1, figsize=(12,8))
    ax.imshow(img)

    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    for box, score in zip(boxes, scores):

        if score < threshold:
            continue

        xmin, ymin, xmax, ymax = box

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )

        ax.add_patch(rect)

        ax.text(
            xmin,
            ymin - 5,
            f"{score:.2f}",
            color="white",
            fontsize=12,
            bbox=dict(facecolor="red", alpha=0.5)
        )

    plt.title("Constellation Detection Result")
    plt.axis("off")
    plt.show()

# =====================================
# 9. RUN DETECTION
# =====================================

detect_uploaded_image(model, device)