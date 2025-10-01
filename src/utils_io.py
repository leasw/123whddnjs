import json, cv2, time
from typing import Optional

class JsonlWriter:
    def __init__(self, path: str, meta: Optional[dict] = None):
        self.f = open(path, "w", encoding="utf-8")
        if meta:
            self.write_line({"session": meta})

    def write_line(self, payload: dict):
        self.f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self):
        self.f.close()

def open_video(source: str):
    try:
        cam_idx = int(source)
        cap = cv2.VideoCapture(cam_idx)
    except ValueError:
        cap = cv2.VideoCapture(source)
    return cap

class Tick:
    def __init__(self):
        self.t = time.time()
    def lap(self):
        now = time.time()
        dt = now - self.t
        self.t = now
        return dt
