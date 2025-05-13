#!/usr/bin/env python

# Imports
from data.labeled_dataset import LabeledDataset
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm
from utils.create_model_dir import create_model_dir
from utils.early_stopping import EarlyStopping
from utils.evaluate_map import evaluate_map
from utils.linear_warmup import linear_warmup
from utils.logger import Logger
from utils.resize_with_aspect import resize_with_aspect
import os
import torch

def get_model(num_classes, device):
    backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=ResNet18_Weights.IMAGENET1K_V1)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model.to(device)

def train_faster_rcnn(model, train_loader, val_loader, device, epochs, logger, warmup_pct=0.05):
    model.to(device)
    model.train()

    base_lr = 5e-4
    init_lr = 1e-5
    weight_decay = 1e-4

    total_steps = len(train_loader) * epochs
    warmup_iters = int(total_steps * warmup_pct)
    warmup_iters = max(10, min(warmup_iters, 500))

    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    early_stopping = EarlyStopping(patience=10, delta=0.001, mode="max")
    global_iter = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Warm-up: linearly increase LR for warmup_iters steps
            if global_iter < warmup_iters:
                lr = linear_warmup(global_iter, warmup_iters, base_lr, init_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            global_iter += 1
        
        scheduler.step()

        # Validation
        val_map = evaluate_map(model, val_loader, device)["map_50"]

        logger.log(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, mAP@0.5: {val_map:.4f}")

        # Early stopping
        early_stopping(val_map, model)
        if early_stopping.early_stop:
            logger.log(f"Early stopping at epoch {epoch+1}")
            break

    return early_stopping.load_best_model(model)

# ====== Safe Entry Point ======
if __name__ == "__main__":
    # Configuration
    PROJECT_NAME = input("Please enter a project name: ")
    UATD_PATH = input("Please enter the path to the FULL UATD-dataset: ")
    DATASET_FRACTION = float(input("Please enter the fraction of the dataset to be used: ") or 1.0)
    SEED = int(input("Please enter the seed: ") or 42)
    BATCH_SIZE = int(input("Please enter the batch size: "))
    EPOCHS = int(input("Please enter the amount of epochs: "))
    WARMUP_PCT = float(input("Please enter the warm-up percentage: ") or 0.05)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 11  # 10 classes + background

    log_path, model_path = create_model_dir(project_name=PROJECT_NAME)
    
    # Logging & timing
    logger = Logger(log_file=log_path)

    logger.log(f"Dataset fraction: {DATASET_FRACTION}", to_console=False)
    logger.log(f"Seed: {SEED}", to_console=False)
    logger.log(f"Batch size: {BATCH_SIZE}", to_console=False)
    logger.log(f"Epochs: {EPOCHS}", to_console=False)
    logger.log(f"Warm-up percentage: {WARMUP_PCT}", to_console=False)

    start_time = datetime.now()

    # Datasets
    train_dataset = LabeledDataset(
        os.path.join(UATD_PATH, "UATD_Training", "images"),
        os.path.join(UATD_PATH, "UATD_Training", "annotations"),
        transforms=resize_with_aspect,
        dataset_fraction=DATASET_FRACTION,
        seed=SEED
    )

    test_dataset = LabeledDataset(
        os.path.join(UATD_PATH, "UATD_Test_1", "images"),
        os.path.join(UATD_PATH, "UATD_Test_1", "annotations"),
        transforms=resize_with_aspect
    )

    val_dataset = LabeledDataset(
        os.path.join(UATD_PATH, "UATD_Test_2", "images"),
        os.path.join(UATD_PATH, "UATD_Test_2", "annotations"),
        transforms=resize_with_aspect
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes=NUM_CLASSES, device=DEVICE)
    model = train_faster_rcnn(model=model, train_loader=train_loader, val_loader=val_loader, device=DEVICE, epochs=EPOCHS, logger=logger, warmup_pct=WARMUP_PCT)

    # Export Model
    torch.save(model.state_dict(), model_path)
    logger.log(f"Model saved at '{model_path}'")

    # Evaluate mAP
    eval_result = evaluate_map(model=model, data_loader=test_loader, device=DEVICE)
    logger.log(eval_result)

    # Final timing
    total_time = datetime.now() - start_time
    logger.log(f"Total Duration: {total_time}")