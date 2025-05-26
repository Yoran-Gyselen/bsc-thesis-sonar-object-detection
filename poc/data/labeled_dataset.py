from PIL import Image
from torchvision.transforms import functional as F
from utils.parse_xml import parse_xml
import os
import random
import torch

class LabeledDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_dir, annotation_dir, transforms=None, dataset_fraction=1.0, seed=42):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms

        all_filenames = sorted(os.listdir(image_dir))
        num_samples = int(len(all_filenames) * dataset_fraction)
        
        # Reproducible random sampling
        random.seed(seed)
        self.image_list = random.sample(all_filenames, num_samples) if dataset_fraction < 1.0 else all_filenames
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        ann_path = os.path.join(self.annotation_dir, self.image_list[idx].replace(".bmp", ".xml"))

        img = Image.open(img_path).convert("RGB")
        boxes, labels = parse_xml(ann_path)

        if self.transforms:
            img, boxes, labels = self.transforms(img, boxes, labels)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if target["boxes"].shape[0] == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.image_list)