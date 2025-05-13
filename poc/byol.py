import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from data.byol_dataset import BYOLDataset
from torch.utils.data import DataLoader
from data.labeled_dataset import LabeledDataset
from data.fixmatch_dataset import FixMatchDataset
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.create_model_dir import create_model_dir
from utils.early_stopping import EarlyStopping
from utils.evaluate_map import evaluate_map
from utils.logger import Logger
from utils.resize_with_aspect import resize_with_aspect
import os
import torch

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
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
        self.encoder = backbone
        self.projector = MLPHead(in_dim=2048) # Adjust based on encoder
        self.predictor = MLPHead(in_dim=256, out_dim=256)
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        z = self.projector(x)
        p = self.predictor(z)
        return p, z.detach()

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
    resnet = models.resnet50()
    resnet.fc = nn.Identity()
    online_net = BYOL(resnet)
    target_net = BYOL(resnet)
    target_net.load_state_dict(online_net.state_dict())
    return online_net.to(device), target_net.to(device)

def train_byol_backbone(train_loader, online_net, target_net, device, epochs=100):
    optimizer = torch.optim.Adam(online_net.parameters(), lr=3e-4)

    # Training loop
    for epoch in range(epochs):
        for view1, view2 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            view1, view2 = view1.to(device), view2.to(device)

            # Online network
            p1, z1 = online_net(view1)
            p2, z2 = online_net(view2)

            # Target network
            with torch.no_grad():
                _, t1 = target_net(view1)
                _, t2 = target_net(view2)
            
            loss = loss_fn(p1, t2) + loss_fn(p2, t1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema(target_net, online_net, beta=0.99)

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
    train_dataset = BYOLDataset(image_dir=os.path.join(UATD_PATH, "UATD_Training", "images"), transforms=train_transform)
    test_dataset = BYOLDataset(image_dir=os.path.join(UATD_PATH, "UATD_Test_1", "images"), transforms=train_transform)
    val_dataset = BYOLDataset(image_dir=os.path.join(UATD_PATH, "UATD_Test_2", "images"), transforms=train_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    online_net, target_net = get_models(DEVICE)
    
    model = load_pretrained_model(num_classes=NUM_CLASSES, model_path=BACKBONE_PATH, device=DEVICE)
    model = train_fixmatch(model=model, data_loader=train_loader, device=DEVICE, logger=logger, epochs=EPOCHS)

    # Export Model
    torch.save(model.state_dict(), model_path)
    logger.log(f"Model saved at '{model_path}'")

    # Evaluate mAP
    eval_result = evaluate_map(model=model, data_loader=test_loader, device=DEVICE)
    logger.log(eval_result)

    # Final timing
    total_time = datetime.now() - start_time
    logger.log(f"Total Duration: {total_time}")