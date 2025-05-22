#!/usr/bin/env python

# Imports
from data.labeled_dataset import LabeledDataset
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm
from utils.create_model_dir import create_model_dir
from utils.evaluate_map import evaluate_map
from utils.logger import Logger
from utils.resize_with_aspect import resize_with_aspect
from utils.early_stopping import EarlyStopping
import os
import torch
from torch.amp import GradScaler, autocast

def get_model(trained_model_path, num_classes, device):
    backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=None)
    backbone.out_channels = 256

    # Load BYOL backbone weights into the FPN body
    state = torch.load(trained_model_path, map_location=device)
    backbone.body.load_state_dict(state, strict=False)

    model = FasterRCNN(backbone, num_classes=num_classes)
    return model.to(device)

def set_bn_train(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.train()

def get_optimizer(model, base_lr, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "BatchNorm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.SGD([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ], lr=base_lr, momentum=0.9)

def finetune_faster_rcnn(
    model, train_loader, val_loader, device, epochs, logger,
    warmup_pct=0.05, unfreeze_epoch=3, validate_before_unfreeze=False
):
    model.to(device)

    # === Freeze early backbone layers ===
    frozen_layers = []
    if hasattr(model.backbone, 'body'):
        for name, param in model.backbone.body.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False
                frozen_layers.append(name)
        logger.log(f"Initially frozen backbone layers: {frozen_layers}")

    base_lr = 0.001
    init_lr = 1e-5
    weight_decay = 1e-4

    optimizer = get_optimizer(model, init_lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=base_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=warmup_pct,
        anneal_strategy='cos',
        final_div_factor=10
    )

    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=10, delta=0.001, mode="max")

    global_iter = 0
    unfrozen = False

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        if epoch == unfreeze_epoch:
            unfrozen = True
            unfrozen_layers = []
            for name, param in model.backbone.body.named_parameters():
                if "layer1" in name or "layer2" in name:
                    param.requires_grad = True
                    unfrozen_layers.append(name)

            # Reset optimizer and scheduler to include new params
            optimizer = get_optimizer(model, base_lr, weight_decay)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=base_lr,
                steps_per_epoch=len(train_loader),
                epochs=epochs - epoch,
                pct_start=0.1,
                anneal_strategy='cos',
                final_div_factor=10
            )
            model.backbone.body.apply(set_bn_train)
            logger.log(f"Unfroze layers at epoch {epoch}: {unfrozen_layers}")

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast(device_type=str(device)):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += losses.item()
            global_iter += 1

        avg_loss = total_loss / len(train_loader)

        # Validation
        run_val = unfrozen or validate_before_unfreeze
        if run_val:
            val_map = evaluate_map(model, val_loader, device)["map_50"]
            logger.log(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}, "
                       f"LR: {optimizer.param_groups[0]['lr']:.6f}, mAP@0.5: {val_map:.4f}")

            early_stopping(val_map, model)
            if early_stopping.early_stop:
                logger.log(f"Early stopping at epoch {epoch+1}")
                break
        else:
            logger.log(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}, "
                       f"LR: {optimizer.param_groups[0]['lr']:.6f}, mAP@0.5: (skipped — frozen)")

    return early_stopping.load_best_model(model)

# ====== Safe Entry Point ======
if __name__ == "__main__":
    # Configuration
    PROJECT_NAME = input("Please enter a project name: ")
    TRAINED_MODEL_PATH = input("Please enter the path to the trained BYOL-backbone: ")
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

    val_dataset = LabeledDataset(
        os.path.join(UATD_PATH, "UATD_Test_2", "images"),
        os.path.join(UATD_PATH, "UATD_Test_2", "annotations"),
        transforms=resize_with_aspect
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(trained_model_path=TRAINED_MODEL_PATH, num_classes=NUM_CLASSES, device=DEVICE)
    model = finetune_faster_rcnn(model=model, train_loader=train_loader, val_loader=val_loader, device=DEVICE, epochs=EPOCHS, logger=logger)
    
    # Export Model
    torch.save(model.state_dict(), model_path)
    logger.log(f"Model saved at '{model_path}'")

    # Evaluate mAP
    eval_result = evaluate_map(model=model, data_loader=test_loader, device=DEVICE)
    logger.log(eval_result)

    # Final timing
    total_time = datetime.now() - start_time
    logger.log(f"Total Duration: {total_time}")