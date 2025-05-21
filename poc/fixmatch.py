#!/usr/bin/env python

# Imports
from data.labeled_dataset import LabeledDataset
from data.fixmatch_dataset import FixMatchDataset
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms.v2 as T
from tqdm import tqdm
from utils.create_model_dir import create_model_dir
from utils.early_stopping import EarlyStopping
from utils.evaluate_map import evaluate_map
from utils.logger import Logger
from utils.resize_with_aspect import resize_with_aspect
import os
import torch
from torch.amp import autocast, GradScaler

def get_weak_transform():
    return T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

def get_strong_transform():
    return T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandAugment(),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

def get_model(num_classes, device):
    backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=ResNet18_Weights.IMAGENET1K_V1)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model.to(device)

def train_fixmatch(
    model, train_loader, val_loader, device, logger,
    min_thresh=0.5, max_thresh=0.95, epochs=10, unfreeze_epoch=3
):
    model.to(device)
    model.train()

    # Separate parameter groups for backbone and head
    def get_param_groups(model):
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        return [
            {"params": backbone_params, "lr": 0.0005},  # smaller LR for pretrained backbone
            {"params": head_params, "lr": 0.005},
        ]

    optimizer = torch.optim.SGD(
        get_param_groups(model),
        momentum=0.9,
        weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=8, delta=0.002, mode="max")
    best_val_map = 0.0  # Track best validation mAP for threshold scheduling

    # Freeze backbone initially
    for param in model.backbone.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        # Optionally unfreeze backbone
        if epoch == unfreeze_epoch:
            logger.log(f"Unfreezing backbone at epoch {epoch}")
            for param in model.backbone.parameters():
                param.requires_grad = True

        # Compute adaptive confidence threshold based on best mAP so far but use a slower ramp up
        ramp = best_val_map ** 0.5  # slows early growth
        current_threshold = min_thresh + (max_thresh - min_thresh) * ramp

        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            (x_l, y_l), x_u = batch

            # Move labeled inputs to device
            x_l = [img.to(device) for img in x_l]
            y_l = [{k: v.to(device) for k, v in t.items()} for t in y_l]

            # Supervised loss
            with autocast(device_type=str(device)):
                loss_dict = model(x_l, y_l)
                loss_l = sum(loss for loss in loss_dict.values())

            # Apply weak and strong augmentations
            x_u_weak = [get_weak_transform()(img).to(device) for img in x_u]
            x_u_strong = [get_strong_transform()(img).to(device) for img in x_u]

            # Generate pseudo-labels
            pseudo_targets = []
            model.eval()
            with torch.no_grad():
                preds = model(x_u_weak)
                min_box_area = 8 * 8  # Filter tiny boxes (< 16x16)
                for pred in preds:
                    boxes = pred['boxes']
                    scores = pred['scores']
                    labels = pred['labels']

                    widths = boxes[:, 2] - boxes[:, 0]
                    heights = boxes[:, 3] - boxes[:, 1]
                    areas = widths * heights

                    mask = (scores > current_threshold) & (areas > min_box_area)
                    if mask.sum() == 0:
                        pseudo_targets.append(None)
                    else:
                        pseudo_targets.append({
                            'boxes': boxes[mask].detach(),
                            'labels': labels[mask].detach()
                        })
            model.train()

            # Filter out images with no valid pseudo-labels
            x_u_filtered = []
            pseudo_filtered = []
            for img, tgt in zip(x_u_strong, pseudo_targets):
                if tgt is not None:
                    x_u_filtered.append(img)
                    pseudo_filtered.append(tgt)

            # Logging stats
            num_pseudo_images = len(pseudo_filtered)
            total_pseudo_boxes = sum(len(t['boxes']) for t in pseudo_filtered)

            # Unsupervised loss
            if x_u_filtered:
                with autocast(device_type=str(device)):
                    loss_dict_u = model(x_u_filtered, pseudo_filtered)
                    loss_u = sum(loss for loss in loss_dict_u.values())
            else:
                loss_u = torch.zeros(1, device=device, requires_grad=True)

            # Adaptive lambda_u based on pseudo-label coverage
            pseudo_ratio = num_pseudo_images / len(x_u)
            lambda_u = min(1.0, max(0.1, pseudo_ratio))

            # Total loss
            with autocast(device_type=str(device)):
                loss = loss_l + lambda_u * loss_u

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Evaluate on validation set
        model.eval()
        val_map = evaluate_map(model, val_loader, device)["map_50"]
        best_val_map = max(best_val_map, val_map)

        # Logging
        logger.log(
            f"Epoch {epoch+1} | "
            f"Total loss: {loss.item():.4f}, "
            f"Labeled loss: {loss_l.item():.4f}, "
            f"Unlabeled loss: {loss_u.item():.4f}, "
            f"mAP@0.5: {val_map.item():.4f}, "
            f"Best mAP: {best_val_map.item():.4f}, "
            f"Threshold: {current_threshold:.3f}, "
            f"lambda_u: {lambda_u:.3f}, "
            f"Pseudo-images: {num_pseudo_images}, "
            f"Pseudo-boxes: {total_pseudo_boxes}"
        )

        # Early stopping
        early_stopping(val_map, model)
        if early_stopping.early_stop:
            logger.log(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    return early_stopping.load_best_model(model)

def fixmatch_collate_fn(batch):
    labeled_imgs = []
    labeled_targets = []
    unlabeled_imgs = []

    for (l_img, target), u_img in batch:
        labeled_imgs.append(l_img)
        labeled_targets.append(target)
        unlabeled_imgs.append(u_img)

    return (labeled_imgs, labeled_targets), unlabeled_imgs

# ====== Safe Entry Point ======
if __name__ == "__main__":
    # Configuration
    PROJECT_NAME = input("Please enter a project name: ")
    UATD_PATH = input("Please enter the path to the FULL UATD-dataset: ")
    LABELED_FRACTION = float(input("Please enter the fraction of the dataset to be used for supervised training: ") or 0.1)
    SEED = int(input("Please enter the seed: ") or 42)
    MIN_THRESH = float(input("Please enter the pseudo-label minimum threshold: ") or 0.5)
    MAX_THRESH = float(input("Please enter the pseudo-label maximum threshold: ") or 0.95)
    BATCH_SIZE = int(input("Please enter the batch size: "))
    EPOCHS = int(input("Please enter the amount of epochs: "))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 11  # 10 classes + background

    log_path, model_path = create_model_dir(project_name=PROJECT_NAME)
    
    # Logging & timing
    logger = Logger(log_file=log_path)

    logger.log(f"Labeled fraction: {LABELED_FRACTION}", to_console=False)
    logger.log(f"Seed: {SEED}", to_console=False)
    logger.log(f"Minimum threshold: {MIN_THRESH}", to_console=False)
    logger.log(f"Maximum threshold: {MAX_THRESH}", to_console=False)
    logger.log(f"Batch size: {BATCH_SIZE}", to_console=False)
    logger.log(f"Epochs: {EPOCHS}", to_console=False)

    start_time = datetime.now()

    # Datasets
    train_dataset = FixMatchDataset(
        os.path.join(UATD_PATH, "UATD_Training", "images"),
        os.path.join(UATD_PATH, "UATD_Training", "annotations"),
        labeled_fraction=LABELED_FRACTION,
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fixmatch_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Model
    model = get_model(num_classes=NUM_CLASSES, device=DEVICE)
    model = train_fixmatch(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=DEVICE, 
        logger=logger,
        min_thresh=MIN_THRESH, 
        max_thresh=MAX_THRESH, 
        epochs=EPOCHS
    )

    # Export Model
    torch.save(model.state_dict(), model_path)
    logger.log(f"Model saved at '{model_path}'")

    # Evaluate mAP
    eval_result = evaluate_map(model=model, data_loader=test_loader, device=DEVICE)
    logger.log(eval_result)

    # Final timing
    total_time = datetime.now() - start_time
    logger.log(f"Total Duration: {total_time}")