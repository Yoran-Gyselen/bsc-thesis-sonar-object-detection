import os
import torch
from torchvision.transforms import functional as F
from PIL import Image

class BYOLDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_dir, transform1, transform2):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.transform1 = transform1
        self.transform2 = transform2
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])

        img = Image.open(img_path).convert("RGB")

        view1 = F.to_tensor(self.transform1(img))
        view2 = F.to_tensor(self.transform2(img))

        return view1, view2

    def __len__(self):
        return len(self.image_list)