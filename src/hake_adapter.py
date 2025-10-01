# src/hake_adapter.py
from typing import List, Dict

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
    return inter/ua

class HakeModel:
    def __init__(self, weight_path: str = None, device: str = None):
        self.ready = True
        self.hold_set = {"cup","bottle","cell phone","wine glass","remote","book"}
        self.ride_set = {"bicycle","motorbike","motorcycle"}

    def infer(self, img_bgr, persons: List[Dict], objects: List[Dict]) -> List[Dict]:
        hoi = []
        for hi, h in enumerate(persons):
            for oi, o in enumerate(objects):
                name = (o.get("category") or "").lower()
                i = _iou(h["bbox"], o["bbox"])
                if name in self.ride_set and i > 0.05:
                    hoi.append({"human_idx":hi,"object_idx":oi,"verb":"ride","score":0.85})
                if name in self.hold_set and i > 0.01:
                    hoi.append({"human_idx":hi,"object_idx":oi,"verb":"hold","score":0.80,"part":"hand","part_score":0.70})
        return hoi
