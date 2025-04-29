import os
import torch
from torchvision.transforms import functional as F
import xml.etree.ElementTree as ET
from PIL import Image

class UATDDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.image_list = sorted(os.listdir(image_dir))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        ann_path = os.path.join(self.annotation_dir, self.image_list[idx].replace(".bmp", ".xml"))

        img = Image.open(img_path).convert("RGB")
        boxes, labels = self.parse_xml(ann_path)

        if self.transforms:
            img, boxes = self.transforms(img, boxes)

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

    def parse_xml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        boxes = []
        labels = []
        label_map = {
            "cube": 1, "ball": 2, "cylinder": 3, "human body": 4,
            "tyre": 5, "square cage": 6, "plane": 7,
            "rov": 8, "circle cage": 9, "metal bucket": 10
        }

        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            b = [int(bbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]
            boxes.append(b)
            labels.append(label_map.get(name, 0))

        return boxes, labels

def resize_with_aspect(image, bboxes, new_height=512):
    old_width, old_height = image.size

    # Calculate scaling factor
    scale = new_height / old_height
    new_width = int(old_width * scale)

    # Resize the image
    resized_image = F.resize(image, [new_height, new_width])

    # Scale bounding boxes
    adjusted_bboxes = []

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox

        x_min = (x_min * new_width) / old_width
        y_min = (y_min * new_height) / old_height
        x_max = (x_max * new_width) / old_width
        y_max = (y_max * new_height) / old_height

        if not (x_max <= x_min or y_max <= y_min):
            adjusted_bboxes.append([x_min, y_min, x_max, y_max])
            # tqdm.write(f"[DEBUG] Removing invalid bounding box | Original: {bbox}, Adjusted: {x_min, y_min, x_max, y_max}")
        # else:
            # adjusted_bboxes.append([x_min, y_min, x_max, y_max])

    return resized_image, adjusted_bboxes