#!/usr/bin/env python

# Imports
from data.byol_dataset import BYOLDataset
from datetime import datetime
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from utils.create_model_dir import create_model_dir
from utils.early_stopping import EarlyStopping
from utils.logger import Logger
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MLPHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projector = MLPHead(in_dim=self.backbone.fc.in_features)
        self.predictor = MLPHead(in_dim=256, out_dim=256)

        # Replace classification head with identity to extract features
        self.backbone.fc = nn.Identity()
    
    def forward(self, x):
        x = self.backbone(x)
        z = self.projector(x)
        p = self.predictor(z)
        return p, z.detach()

@torch.no_grad()
def update_ema(target_net, online_net, beta=0.99):
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data = beta * target_param.data + (1. - beta) * online_param.data

def loss_fn(x, y):
   # L2 normalization
   x = F.normalize(x, dim=-1)
   y = F.normalize(y, dim=-1)
   return 2 - 2 * (x * y).sum(dim=-1).mean()

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])

def get_models(device):
    backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    online_net = BYOL(backbone).to(device)
    target_net = BYOL(models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)).to(device)
    target_net.load_state_dict(online_net.state_dict())

    # Freeze target network parameters
    for param in target_net.parameters():
        param.requires_grad = False

    return online_net, target_net

def train_byol_backbone(train_loader, online_net, target_net, logger, device, epochs=100):
    optimizer = torch.optim.Adam(online_net.parameters(), lr=3e-4)
    early_stopping = EarlyStopping(patience=10, delta=0.001, mode="min")
    scaler = GradScaler()

    for epoch in range(epochs):
        running_loss = 0
        online_net.train()

        for view1, view2 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()

            with autocast(device_type=str(device)):
                # Online network forward
                p1, z1 = online_net(view1)
                p2, z2 = online_net(view2)

                # Target network forward
                with torch.no_grad():
                    _, t1 = target_net(view1)
                    _, t2 = target_net(view2)

                loss = loss_fn(p1, t2) + loss_fn(p2, t1)

            # Backpropagation with GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_ema(target_net, online_net, beta=0.99)
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logger.log(f"Epoch {epoch+1}/{epochs} | Average loss: {avg_loss:.4f}")

        # Early stopping check
        early_stopping(avg_loss, online_net.backbone)
        if early_stopping.early_stop:
            logger.log(f"Early stopping at epoch {epoch+1}")
            break

    return early_stopping.load_best_model(online_net.backbone)

# ====== Safe Entry Point ======
if __name__ == "__main__":
    # Configuration
    PROJECT_NAME = input("Please enter a project name: ")
    UATD_PATH = input("Please enter the path to the FULL UATD-dataset: ")
    BATCH_SIZE = int(input("Please enter the batch size: "))
    EPOCHS = int(input("Please enter the amount of epochs: "))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 11  # 10 classes + background

    log_path, model_path = create_model_dir(project_name=PROJECT_NAME)
    
    # Logging & timing
    logger = Logger(log_file=log_path)

    logger.log(f"Batch size: {BATCH_SIZE}", to_console=False)
    logger.log(f"Epochs: {EPOCHS}", to_console=False)

    start_time = datetime.now()

    # Datasets
    train_dataset = BYOLDataset(image_dir=os.path.join(UATD_PATH, "UATD_Training", "images"), transform1=train_transform, transform2=train_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Models
    online_net, target_net = get_models(DEVICE)
    online_net_backbone = train_byol_backbone(train_loader=train_loader, online_net=online_net, target_net=target_net, logger=logger, device=DEVICE, epochs=EPOCHS)

    # Export Model
    torch.save(online_net_backbone.state_dict(), model_path)
    logger.log(f"Model saved at '{model_path}'")

    # Final timing
    total_time = datetime.now() - start_time
    logger.log(f"Total Duration: {total_time}")