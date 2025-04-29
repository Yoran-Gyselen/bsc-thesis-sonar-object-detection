import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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