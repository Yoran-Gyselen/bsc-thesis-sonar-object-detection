import os
import torch
import random
from utils.parse_xml import parse_xml
from utils.resize_with_aspect import resize_with_aspect, resize_img_with_aspect
from torchvision.transforms import functional as F
from PIL import Image
from itertools import cycle

class FixMatchDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_dir, annotation_dir, labeled_fraction=0.1, seed=42):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

        all_filenames = sorted(os.listdir(image_dir))
        num_labeled_samples = int(len(all_filenames) * labeled_fraction)
        
        # Reproducible random sampling
        random.seed(seed)
        self.labeled_image_list = random.sample(all_filenames, num_labeled_samples) if labeled_fraction < 1.0 else all_filenames
        self.unlabeled_image_list = [img for img in all_filenames if img not in self.labeled_image_list]
        self.labeled_iter = cycle(self.labeled_image_list)
    
    def __getitem__(self, idx):
        # Get paths
        unlabeled_img_path = os.path.join(self.image_dir, self.unlabeled_image_list[idx])

        labeled_img_name = next(self.labeled_iter)
        labeled_img_path = os.path.join(self.image_dir, labeled_img_name)
        labeled_ann_path = os.path.join(self.annotation_dir, labeled_img_name.replace(".bmp", ".xml"))

        # Open the paths
        unlabeled_img = Image.open(unlabeled_img_path).convert("RGB")
        labeled_img = Image.open(labeled_img_path).convert("RGB")
        boxes, labels = parse_xml(labeled_ann_path)

        # Resize while keeping aspect ratio
        labeled_img, boxes, labels = resize_with_aspect(labeled_img, boxes, labels)
        unlabeled_img, _, _ = resize_img_with_aspect(unlabeled_img)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if target["boxes"].shape[0] == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        labeled_img = F.to_tensor(labeled_img)
        unlabeled_img = F.to_tensor(unlabeled_img)

        return (labeled_img, target), unlabeled_img

    def __len__(self):
        return len(self.unlabeled_image_list)