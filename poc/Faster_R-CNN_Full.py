# ====== Imports ======
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import box_iou
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# ====== Input and output path ======
UATD_PATH = input("Please enter the path to the FULL UATD-dataset: ")
assert os.path.exists(UATD_PATH), "UATD-dataset path does not exist"

OUTPUT_PATH = input("Please enter the path of the directory where the model should be saved: ")
assert os.path.exists(OUTPUT_PATH), "Output path does not exist"

# ====== Start timing ======
start_time = datetime.now()

# ====== Configuration ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11  # 10 classes + background
BATCH_SIZE = int(input("Please enter the batch size: "))
EPOCHS = int(input("Please enter the amount of epochs: "))

# ====== UATDDataset class ======
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

        x_min = round((x_min * new_width) / old_width)
        y_min = round((y_min * new_height) / old_height)
        x_max = round((x_max * new_width) / old_width)
        y_max = round((y_max * new_height) / old_height)

        adjusted_bboxes.append([x_min, y_min, x_max, y_max])

    return image, adjusted_bboxes

# ====== Datasets ======
# Datasets
train_dataset = UATDDataset(
    os.path.join(UATD_PATH, "UATD_Training", "images"),
    os.path.join(UATD_PATH, "UATD_Training", "annotations"),
    transforms=resize_with_aspect
)

test_dataset = UATDDataset(
    os.path.join(UATD_PATH, "UATD_Test_1", "images"),
    os.path.join(UATD_PATH, "UATD_Test_1", "annotations"),
    transforms=resize_with_aspect
)

# ====== DataLoaders =======
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=lambda x: tuple(zip(*x))
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# ====== Model ======
backbone = resnet_fpn_backbone("resnet18", pretrained=True)
model = FasterRCNN(backbone, num_classes=NUM_CLASSES)
model.to(DEVICE)

# ====== Optimizer & Scheduler ======
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)

# ====== Training Loop ======
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    print(f"Loss: {total_loss/len(train_loader):.4f}")

    scheduler.step()

print(f"Training Duration: {datetime.now() - start_time}")

# ====== Evaluate mAP-like metric ======
def evaluate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                # Example: simple IoU print
                pred_boxes = output["boxes"].cpu()
                true_boxes = target["boxes"]
                iou = box_iou(pred_boxes, true_boxes)
                print(f"IoU matrix:\n{iou}")

evaluate(model, test_loader)

# ====== Export Model ======
model_name = "fasterrcnn_uatd_full.pth"
model_path = os.path.join(OUTPUT_PATH, model_name)

torch.save(model.state_dict(), model_path)
print(f"Model saved at '{model_path}'")

print(f"Final Duration: {datetime.now() - start_time}")
