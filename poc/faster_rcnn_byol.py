#!/usr/bin/env python

# Imports
from data.labeled_dataset import LabeledDataset
from datetime import datetime
from torch.utils.data import DataLoader
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

def get_model(trained_model_path, num_classes, device):
    backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=None)
    backbone.out_channels = 256

    # Load BYOL backbone weights into the FPN body
    state = torch.load(trained_model_path, map_location=device)
    backbone.body.load_state_dict(state, strict=False)

    model = FasterRCNN(backbone, num_classes=num_classes)
    return model.to(device)

def finetune_faster_rcnn(model, train_loader, val_loader, device, epochs, logger, warmup_pct=0.05, unfreeze_epoch=3):
    model.to(device)
    model.train()

    # === Freeze early backbone layers and BatchNorm layers ===
    if hasattr(model.backbone, 'body'):
        frozen = []
        for name, param in model.backbone.body.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False
                frozen.append(name)

        logger.log(f"Initially frozen backbone layers: {frozen}")

    base_lr = 0.001
    init_lr = 1e-5
    weight_decay = 1e-4

    total_steps = len(train_loader) * epochs
    warmup_iters = max(10, min(int(total_steps * warmup_pct), 500))

    def get_optimizer():
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.SGD(params, lr=init_lr, momentum=0.9, weight_decay=weight_decay)

    optimizer = get_optimizer()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopping = EarlyStopping(patience=10, delta=0.001, mode="max")

    global_iter = 0
    train_loss_history = []
    unfrozen_state = False

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # Unfreeze at specified epoch
        if epoch == unfreeze_epoch:
            unfrozen_state = True
            unfrozen_layers = []
            for name, param in model.backbone.body.named_parameters():
                if "layer1" in name or "layer2" in name:
                    param.requires_grad = True
                    unfrozen_layers.append(name)
            optimizer = get_optimizer()  # Rebuild optimizer
            logger.log(f"Unfroze backbone layers at epoch {epoch}: {unfrozen_layers}")


        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Warm-up
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

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        scheduler.step()

        # Validation & Early stopping
        if unfrozen_state:
            val_map = evaluate_map(model, val_loader, device)["map_50"]
            logger.log(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}, mAP@0.5: {val_map:.4f}")

            early_stopping(val_map, model)
            if early_stopping.early_stop:
                logger.log(f"Early stopping at epoch {epoch+1}")
                break
        else:
            logger.log(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}, mAP@0.5: (skipped — frozen)"
            )
 

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