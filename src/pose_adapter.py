# src/pose_adapter.py
from typing import List, Dict, Optional
import numpy as np
from ultralytics import YOLO

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
    return inter / ua

class PoseEstimator:
    """
    Ultralytics pose model wrapper (e.g., yolov8n-pose.pt).
    Fills persons[i]["keypoints"] = [[x,y,conf], ...] in COCO order.
    """
    def __init__(self, weight_path: str, conf: float = 0.25, iou: float = 0.5, device: Optional[str] = None):
        self.model = YOLO(weight_path)
        self.conf = conf
        self.iou = iou
        self.device = device

    def __call__(self, img_bgr, persons: List[Dict]) -> List[Dict]:
        if not persons:
            return persons
        res = self.model.predict(img_bgr, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return persons

        # pose det outputs
        pose_boxes = res.boxes.xyxy.cpu().numpy()
        pose_kp = res.keypoints  # shape: [N, K, 2] or [N, K, 3]
        kp_np = pose_kp.data.cpu().numpy()  # [N, K, D], D=2 or 3

        # match each pose-box to tracked persons by IoU
        for pi in range(len(pose_boxes)):
            pb = pose_boxes[pi].tolist()
            kps = kp_np[pi]
            # ensure [x,y,conf]
            if kps.shape[1] == 2:
                kps = np.concatenate([kps, np.ones((kps.shape[0], 1), dtype=kps.dtype)], axis=1)
            best, best_j = 0.0, -1
            for j, p in enumerate(persons):
                iou = _iou_xyxy(pb, p["bbox"])
                if iou > best:
                    best, best_j = iou, j
            if best_j >= 0 and best > 0.1:
                persons[best_j]["keypoints"] = kps.tolist()
        return persons
