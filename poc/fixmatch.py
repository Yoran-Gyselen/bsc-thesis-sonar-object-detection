#!/usr/bin/env python

# Imports
from data.labeled_dataset import LabeledDataset
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms.v2 as T
from tqdm import tqdm
from utils.create_model_dir import create_model_dir
from utils.early_stopping import EarlyStopping
from utils.evaluate_map import evaluate_map
from utils.linear_warmup import linear_warmup
from utils.logger import Logger
from utils.resize_with_aspect import resize_with_aspect
import os
import torch

def get_weak_transform():
    return T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(p=0.5),
    ])

def get_strong_transform():
    return T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandAugment(),  # Strong augmentations
    ])

def load_pretrained_model(num_classes, model_path, device):
    # Create a fresh model instance with correct output classes
    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=ResNet50_Weights.IMAGENET1K_V2)
    model = FasterRCNN(backbone, num_classes=num_classes)

    # Load your 10%-trained weights
    model.load_state_dict(torch.load(model_path))

    return model.to(device)

def train_fixmatch(model, data_loader, device, logger, threshold=0.95, lambda_u=1.0, epochs=10, unfreeze_epoch=3):
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),  # important!
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    # Freeze backbone before training
    for param in model.backbone.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            logger.log(f"Unfreezing backbone at epoch {epoch}")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Recreate optimizer with all parameters
            optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        
        for (x_l, y_l), x_u in data_loader:
            x_l = [img.to(device) for img in x_l]
            y_l = [{k: v.to(device) for k, v in t.items()} for t in y_l]

            loss_dict = model(x_l, y_l)
            loss_l = sum(loss for loss in loss_dict.values())

            x_u_weak = [get_weak_transform()(img)[0].to(device) for img in x_u]
            x_u_strong = [get_strong_transform()(img)[0].to(device) for img in x_u]

            pseudo_targets = []
            with torch.no_grad():
                preds = model(x_u_weak)
                for pred in preds:
                    mask = pred['scores'] > threshold
                    if mask.sum() == 0:
                        pseudo_targets.append(None)
                        continue
                    pseudo_targets.append({
                        'boxes': pred['boxes'][mask].detach(),
                        'labels': pred['labels'][mask].detach()
                    })

            x_u_filtered = []
            pseudo_filtered = []
            for img, tgt in zip(x_u_strong, pseudo_targets):
                if tgt is not None:
                    x_u_filtered.append(img)
                    pseudo_filtered.append(tgt)

            if x_u_filtered:
                loss_dict_u = model(x_u_filtered, pseudo_filtered)
                loss_u = sum(loss for loss in loss_dict_u.values())
            else:
                loss_u = torch.tensor(0.0, device=device)

            loss = loss_l + lambda_u * loss_u
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.log(f"Epoch {epoch+1}/{epochs} | Total loss: {loss.item():.4f}, Labeled loss: {loss_l.item():.4f}, Unlabeled loss: {loss_u.item():.4f}")

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
        transforms=resize_with_aspect,
        dataset_fraction=DATASET_FRACTION,
        seed=SEED
    )

    val_dataset = LabeledDataset(
        os.path.join(UATD_PATH, "UATD_Test_2", "images"),
        os.path.join(UATD_PATH, "UATD_Test_2", "annotations"),
        transforms=resize_with_aspect,
        dataset_fraction=DATASET_FRACTION,
        seed=SEED
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes=NUM_CLASSES, device=DEVICE)
    train_faster_rcnn(model=model, train_loader=train_loader, val_loader=val_loader, device=DEVICE, epochs=EPOCHS, logger=logger, warmup_pct=WARMUP_PCT)

    # Export Model
    torch.save(model.state_dict(), model_path)
    logger.log(f"Model saved at '{model_path}'")

    # Evaluate mAP
    eval_result = evaluate_map(model=model, data_loader=test_loader, device=DEVICE)
    logger.log(eval_result)

    # Final timing
    total_time = datetime.now() - start_time
    logger.log(f"Total Duration: {total_time}")