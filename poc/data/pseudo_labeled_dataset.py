import os
import torch
import random
from utils.parse_xml import parse_xml
from torchvision.transforms import functional as F
from PIL import Image

class PseudoLabeledDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_dir, image_filenames, model, device, score_threshold=0.5, transforms=None):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.model = model.eval().to(device)
        self.device = device
        self.transforms = transforms
        self.score_threshold = score_threshold
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            img, boxes, labels = self.transforms(img, boxes, labels)
        
        with torch.no_grad():
            output = self.model([img.to(self.device)])[0]
        
        # Filter predictions by score
        keep = output["scores"] > self.score_threshold
        pseudo_boxes = output["boxes"][keep].cpu()
        pseudo_labels = output["labels"][keep].cpu()

        target = {
            "boxes": torch.tensor(pseudo_boxes, dtype=torch.float32),
            "labels": torch.tensor(pseudo_labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if target["boxes"].shape[0] == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.image_filenames)