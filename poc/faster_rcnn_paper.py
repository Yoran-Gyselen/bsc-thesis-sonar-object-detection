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

def train_faster_rcnn(model, train_loader, device, epochs, logger):
    model.to(device)
    model.train()

    base_lr = 5e-4
    warmup_iters = 500
    warmup_start_lr = 1e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 10], gamma=0.1)

    global_iter = 0

    # Training Loop
    for epoch in range(epochs):
        total_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Warm-up logic
            if global_iter < warmup_iters:
                lr = linear_warmup(global_iter, warmup_iters, base_lr, init_lr=warmup_start_lr)
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

        logger.log(f"Epoch {epoch+1}/{epochs} | Training loss: {total_loss/len(train_loader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return model

# ====== Safe Entry Point ======
if __name__ == "__main__":
    # Configuration
    PROJECT_NAME = input("Please enter a project name: ")
    UATD_PATH = input("Please enter the path to the FULL UATD-dataset: ")
    DATASET_FRACTION = float(input("Please enter the fraction of the dataset to be used: ") or 1.0)
    SEED = int(input("Please enter the seed: ") or 42)
    BATCH_SIZE = int(input("Please enter the batch size: "))
    EPOCHS = int(input("Please enter the amount of epochs: "))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 11  # 10 classes + background
    
    log_path, model_path = create_model_dir(project_name=PROJECT_NAME)
    
    # Logging & timing
    logger = Logger(log_file=log_path)

    logger.log(f"Dataset fraction: {DATASET_FRACTION}", to_console=False)
    logger.log(f"Seed: {SEED}", to_console=False)
    logger.log(f"Batch size: {BATCH_SIZE}", to_console=False)
    logger.log(f"Epochs: {EPOCHS}", to_console=False)

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

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes=NUM_CLASSES, device=DEVICE)
    model = train_faster_rcnn(model=model, train_loader=train_loader, device=DEVICE, epochs=EPOCHS, logger=logger)
    
    # Export Model
    torch.save(model.state_dict(), model_path)
    logger.log(f"Model saved at '{model_path}'")

    # Evaluate mAP
    eval_result = evaluate_map(model=model, data_loader=test_loader, device=DEVICE)
    logger.log(eval_result)

    # Final timing
    total_time = datetime.now() - start_time
    logger.log(f"Total Duration: {total_time}")