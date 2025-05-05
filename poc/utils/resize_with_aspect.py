from torchvision.transforms import functional as F

def resize_img_with_aspect(image, new_height=512):
    old_width, old_height = image.size

    # Calculate scaling factor
    scale = new_height / old_height
    new_width = int(old_width * scale)

    # Resize the image
    resized_image = F.resize(image, [new_height, new_width])

    return resized_image, (old_width, old_height), (new_width, new_height)

def resize_bboxes_with_aspect(old_dims, new_dims, bboxes, labels, new_height=512):
    old_width, old_height = old_dims
    new_width, _ = new_dims

    # Scale bounding boxes
    adjusted_bboxes = []
    adjusted_labels = []

    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = bbox

        x_min = (x_min * new_width) / old_width
        y_min = (y_min * new_height) / old_height
        x_max = (x_max * new_width) / old_width
        y_max = (y_max * new_height) / old_height

        if not (x_max <= x_min or y_max <= y_min):
            adjusted_bboxes.append([x_min, y_min, x_max, y_max])
            adjusted_labels.append(label)

    return adjusted_bboxes, adjusted_labels

def resize_with_aspect(image, boxes, labels, new_height=512):
    resized_image, old_dims, new_dims = resize_img_with_aspect(image=image, new_height=new_height)
    adjusted_bboxes, adjusted_labels = resize_bboxes_with_aspect(old_dims=old_dims, new_dims=new_dims, bboxes=boxes, labels=labels)

    return resized_image, adjusted_bboxes, adjusted_labels