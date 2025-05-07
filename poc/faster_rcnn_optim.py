#!/usr/bin/env python

# ====== Imports ======
import os
import torch
from data.labeled_dataset import LabeledDataset
from datetime import datetime
from utils.resize_with_aspect import resize_with_aspect
from utils.evaluate_map import evaluate_map
from utils.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm
from pathlib import Path

def main():
    # ====== Configuration ======
    PROJECT_NAME = input("Please enter a project name: ")
    UATD_PATH = input("Please enter the path to the FULL UATD-dataset: ")
    DATASET_FRACTION = float(input("Please enter the fraction of the dataset to be used: ") or 1.0)
    SEED = int(input("Please enter the seed: ") or 42)
    BATCH_SIZE = int(input("Please enter the batch size: "))
    EPOCHS = int(input("Please enter the amount of epochs: "))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 11  # 10 classes + background
    
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    WORKING_DIR = os.path.join("models", PROJECT_NAME)
    Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)
    LOG_PATH = os.path.join(WORKING_DIR, f"{PROJECT_NAME}_{TIMESTAMP}.txt")
    MODEL_PATH = LOG_PATH.replace(".txt", ".pth")
    
    # Timing
    start_time = datetime.now()

    # ====== Dataset & DataLoader ======
    train_dataset = LabeledDataset(
        os.path.join(UATD_PATH, "UATD_Training", "images"),
        os.path.join(UATD_PATH, "UATD_Training", "annotations"),
        transforms=resize_with_aspect,
        dataset_fraction=DATASET_FRACTION,
        seed=SEED
    )

    val_dataset = LabeledDataset(
        os.path.join(UATD_PATH, "UATD_Test_1", "images"),
        os.path.join(UATD_PATH, "UATD_Test_1", "annotations"),
        transforms=resize_with_aspect,
        dataset_fraction=DATASET_FRACTION,
        seed=SEED
    )

    test_dataset = LabeledDataset(
        os.path.join(UATD_PATH, "UATD_Test_2", "images"),
        os.path.join(UATD_PATH, "UATD_Test_2", "annotations"),
        transforms=resize_with_aspect,
        dataset_fraction=DATASET_FRACTION,
        seed=SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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

    # ====== Model Setup ======
    backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=ResNet18_Weights.DEFAULT)
    model = FasterRCNN(backbone, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=0.1, patience=5)
    early_stopping = EarlyStopping(patience=10, delta=0.001, mode="max")

    with open(LOG_PATH, "w") as logfile:
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
            
            learning_rate = scheduler.get_last_lr()[0]

            # Validation mAP
            val_map = evaluate_map(model, val_loader, DEVICE)["map_50"]

            # Check early stopping
            early_stopping(val_map, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                logfile.write(f"Early stopping at epoch {epoch+1}\n")
                break  # Stop training if early stopping is triggered

            scheduler.step(val_map)

            print(f"Training loss: {total_loss/len(train_loader):.4f}, LR: {learning_rate}, mAP@0.5: {val_map:.4f}")
            logfile.write(f"Epoch {epoch+1}/{EPOCHS} | Training loss: {total_loss/len(train_loader):.4f}, LR: {learning_rate}, mAP@0.5: {val_map:.4f}\n")
        
        # ====== Export Model ======
        early_stopping.load_best_model(model)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved at '{MODEL_PATH}'")

        # ====== Evaluate mAP ======
        eval_result = evaluate_map(model=model, data_loader=test_loader, device=DEVICE)
        print(f"mAP@0.5: {eval_result['map_50']:.4f}")
        logfile.write(f"{eval_result}\n")

        # ====== Final timing ======
        total_time = datetime.now() - start_time
        print(f"Total Duration: {total_time}")
        logfile.write(f"Total Duration: {total_time}\n")

# ====== Safe Entry Point ======
if __name__ == "__main__":
    main()