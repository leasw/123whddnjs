import numpy as np
from typing import List, Dict

# Try to import the new API explicitly; fall back to top-level if needed
try:
    from supervision.tracker.byte_tracker import ByteTrack, ByteTrackConfig
    _NEW_API = True
except Exception:
    import supervision as sv
    ByteTrack = getattr(sv, "ByteTrack")
    ByteTrackConfig = getattr(sv, "ByteTrackConfig", None)
    _NEW_API = ByteTrackConfig is not None

# Detections type (handle both import styles)
try:
    from supervision import Detections
except Exception:
    # very old versions
    from supervision.detection.core import Detections


def items_to_dets(items: List[Dict]) -> "Detections":
    if not items:
        return Detections.empty()
    xyxy = np.array([it["bbox"] for it in items], dtype=np.float32)
    conf = np.array([it.get("score", 1.0) for it in items], dtype=np.float32)
    cls  = np.array([it.get("class_id", -1) for it in items], dtype=np.int32)
    return Detections(xyxy=xyxy, confidence=conf, class_id=cls)


def dets_to_items(dets: "Detections", orig_items: List[Dict], id_offset: int = 0) -> List[Dict]:
    out = []
    n = len(dets)
    for i in range(n):
        box = dets.xyxy[i].tolist()
        score = float(dets.confidence[i]) if dets.confidence is not None else float(orig_items[i].get("score", 1.0))
        cls_id = int(dets.class_id[i]) if dets.class_id is not None else int(orig_items[i].get("class_id", -1))
        cat = orig_items[i].get("category", f"class_{cls_id}")
        tid = int(dets.tracker_id[i]) + id_offset if dets.tracker_id is not None else None
        out.append({"bbox": box, "score": score, "class_id": cls_id, "category": cat, "track_id": tid})
    return out


def _make_bytrack(track_thresh=0.25, match_thresh=0.8, track_buffer=30) -> "ByteTrack":
    """
    Create a ByteTrack instance compatible with both new (config-based)
    and old (kwargs) supervision APIs.
    """
    # New API: requires ByteTrackConfig
    if _NEW_API and ByteTrackConfig is not None:
        cfg = ByteTrackConfig(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer
        )
        return ByteTrack(cfg)

    # Old API: accepts kwargs directly on constructor
    try:
        return ByteTrack(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer
        )
    except TypeError:
        # As a last resort, default constructor
        return ByteTrack()


def _bt_update(tracker: "ByteTrack", dets: "Detections") -> "Detections":
    """
    Call the appropriate update method depending on supervision version.
    """
    if hasattr(tracker, "update_with_detections"):
        return tracker.update_with_detections(detections=dets)  # older API
    # newer API
    return tracker.update(detections=dets)


class ByteTrackPair:
    """
    Two independent ByteTrack instances:
      - one for humans
      - one for objects
    We offset object IDs by a large constant to avoid collisions with human IDs.
    """
    def __init__(self, track_thresh=0.25, match_thresh=0.8, track_buffer=30, object_id_offset=100000):
        self.h = _make_bytrack(track_thresh, match_thresh, track_buffer)
        self.o = _make_bytrack(track_thresh, match_thresh, track_buffer)
        self.object_id_offset = object_id_offset

    def step(self, persons: List[Dict], objects: List[Dict]):
        h_dets = items_to_dets(persons)
        o_dets = items_to_dets(objects)

        h_out = _bt_update(self.h, h_dets)
        o_out = _bt_update(self.o, o_dets)

        persons = dets_to_items(h_out, persons, id_offset=0)
        objects = dets_to_items(o_out, objects, id_offset=self.object_id_offset)
        return persons, objects
