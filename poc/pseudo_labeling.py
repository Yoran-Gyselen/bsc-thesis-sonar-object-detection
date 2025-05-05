# train/train_pseudo_labeling.py
from config.pseudo_labeling import *
from data.dataloader_factory import get_dataloaders
from models.faster_rcnn import get_model
from utils.metrics import evaluate_map

# 1. Load dataloader using LABEL_FRACTION (e.g. 0.1)
train_loader, test_loader = get_dataloaders(UATD_PATH, BATCH_SIZE, fraction=LABEL_FRACTION)

# 2. Train and save model (same as your full supervision loop)


# Load full image list
all_images = sorted(os.listdir(train_image_dir))

# Load images used in the labeled dataset
used_images = set(train_dataset.image_filenames)  # from step 1
unlabeled_images = [img for img in all_images if img not in used_images]

# Load pretrained model from step 1
model.load_state_dict(torch.load("saved_models/pseudo_step1_initial_model.pth"))

# Create pseudo-labeled dataset
pseudo_dataset = PseudoLabeledDataset(
    image_dir=train_image_dir,
    image_filenames=unlabeled_images,
    model=model,
    device=DEVICE,
    transforms=your_transform_function
)

# Combine with original labeled dataset
from torch.utils.data import ConcatDataset
combined_dataset = ConcatDataset([train_dataset, pseudo_dataset])

# Train again
combined_loader = DataLoader(
    combined_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    collate_fn=lambda x: tuple(zip(*x)),
    pin_memory=True,
    persistent_workers=True
)

# Train model on combined_loader (can reuse training loop)
