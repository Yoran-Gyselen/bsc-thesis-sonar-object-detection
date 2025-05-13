import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

def evaluate_map(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], box_format="xyxy")

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = [{
                "boxes": o["boxes"].cpu(),
                "scores": o["scores"].cpu(),
                "labels": o["labels"].cpu()
            } for o in outputs]

            tgts = [{
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu()
            } for t in targets]

            metric.update(preds, tgts)

    return metric.compute()
