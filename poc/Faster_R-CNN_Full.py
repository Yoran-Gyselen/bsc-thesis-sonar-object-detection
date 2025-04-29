#!/usr/bin/env python

# ====== Imports ======
import os
import torch
from utils.UATDDataset import UATDDataset, resize_with_aspect
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models import ResNet18_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# ====== Input and output path ======
UATD_PATH = input("Please enter the path to the FULL UATD-dataset: ")
assert os.path.exists(UATD_PATH), "UATD-dataset path does not exist"

OUTPUT_PATH = input("Please enter the path of the directory where the model should be saved: ")
assert os.path.exists(OUTPUT_PATH), "Output path does not exist"

# Create logfile
PROJECT_NAME = "faster_rcnn_uatd_full"
WORKING_DIR = os.path.join(OUTPUT_PATH, f"{PROJECT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)

logfile = open(os.path.join(WORKING_DIR, f"{PROJECT_NAME}.log"), "w")

# ====== Start timing ======
start_time = datetime.now()

# ====== Configuration ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11  # 10 classes + background
BATCH_SIZE = int(input("Please enter the batch size: "))
EPOCHS = int(input("Please enter the amount of epochs: "))

# ====== Datasets ======
# Datasets
train_dataset = UATDDataset(
    os.path.join(UATD_PATH, "UATD_Training", "images"),
    os.path.join(UATD_PATH, "UATD_Training", "annotations"),
    transforms=resize_with_aspect
)

test_dataset = UATDDataset(
    os.path.join(UATD_PATH, "UATD_Test_1", "images"),
    os.path.join(UATD_PATH, "UATD_Test_1", "annotations"),
    transforms=resize_with_aspect
)

# ====== DataLoaders =======
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=lambda x: tuple(zip(*x))
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# ====== Model ======
backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=ResNet18_Weights.DEFAULT)
model = FasterRCNN(backbone, num_classes=NUM_CLASSES)
model.to(DEVICE)

# ====== Optimizer & Scheduler ======
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)

# ====== Training Loop ======
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    print(f"Training loss: {total_loss/len(train_loader):.4f}")
    logfile.write(f"Epoch {epoch+1}/{EPOCHS} | Training loss: {total_loss/len(train_loader):.4f}\n")

    scheduler.step()

print(f"Training Duration: {datetime.now() - start_time}")
logfile.write(f"Training Duration: {datetime.now() - start_time}\n")

# ====== Export Model ======
model_name = f"{PROJECT_NAME}.pth"
model_path = os.path.join(WORKING_DIR, model_name)

torch.save(model.state_dict(), model_path)
print(f"Model saved at '{model_path}'")

# ====== Evaluate mAP ======
def evaluate_map(model, data_loader):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            # Convert predictions & targets for torchmetrics
            preds = []
            tgts = []

            for output, target in zip(outputs, targets):
                preds.append({
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu()
                })
                tgts.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu()
                })

            metric.update(preds, tgts)

    final_result = metric.compute()
    print(f"mAP@0.5: {final_result['map_50']:.4f}")
    return final_result

evaluation_result = evaluate_map(model=model, data_loader=test_loader)
logfile.write(f"{evaluation_result}\n")

print(f"Final Duration: {datetime.now() - start_time}")
logfile.write(f"Final Duration: {datetime.now() - start_time}\n")

# Close logfile
logfile.close()