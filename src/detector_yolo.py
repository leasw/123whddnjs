import numpy as np
from ultralytics import YOLO

COCO_PERSON_ID = 0  # Ultralytics: 'person' class index in COCO (usually 0)

class YoloDetector:
    """
    Adapter for Ultralytics YOLOv11/12 detectors.
    Returns:
      persons: [{"bbox":[x1,y1,x2,y2], "score":float}]
      objects: [{"bbox":[x1,y1,x2,y2], "score":float, "category":str}]
    """
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.5, device: str = None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device

        # name map
        self.names = self.model.names if hasattr(self.model, "names") else None

    def __call__(self, img_bgr):
        res = self.model.predict(
            source=img_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )[0]

        persons = []
        objects = []

        if res.boxes is None or len(res.boxes) == 0:
            return persons, objects

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)

        for b, s, c in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = [float(v) for v in b]
            if c == COCO_PERSON_ID:
                persons.append({"bbox": [x1, y1, x2, y2], "score": float(s)})
            else:
                name = self.names[c] if self.names and c in self.names else f"class_{c}"
                objects.append({"bbox": [x1, y1, x2, y2], "score": float(s), "category": name})

        return persons, objects
