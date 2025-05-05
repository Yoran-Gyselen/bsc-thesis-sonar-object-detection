import xml.etree.ElementTree as ET

def parse_xml(path):
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