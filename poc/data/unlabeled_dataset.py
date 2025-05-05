import os
import torch
import random
from torchvision.transforms import functional as F
from PIL import Image

class UnlabeledDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_dir, transforms=None, label_fraction=1.0, seed=42):
        self.image_dir = image_dir
        self.transforms = transforms

        all_filenames = sorted(os.listdir(image_dir))
        num_samples = int(len(all_filenames) * label_fraction)
        
        # Reproducible random sampling
        random.seed(seed)
        self.image_list = random.sample(all_filenames, num_samples) if label_fraction < 1.0 else all_filenames
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])

        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            img, _, _ = self.transforms(img)

        img = F.to_tensor(img)

        return img

    def __len__(self):
        return len(self.image_list)