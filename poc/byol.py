import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from data.byol_dataset import BYOLDataset
from torch.utils.data import DataLoader

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

train_dataset = BYOLDataset(image_dir=, transforms=train_transform)
test_dataset = BYOLDataset(image_dir=, transforms=train_transform)
val_dataset = BYOLDataset(image_dir=, transforms=train_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# Networks
resnet = models.resnet50()
resnet.fc = nn.Identity()
online_net = BYOL(resnet)
target_net = BYOL(resnet)
target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=3e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
online_net.to(device)
target_net.to(device)

# Training loop
for epoch in range(100):
    for view1, view2 in dataloader:
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