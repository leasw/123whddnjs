#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py
YOLOv11/12 (Ultralytics) -> ByteTrack (supervision) -> HAKE (heuristic or real)
+ Visualization: human/object boxes+IDs, verbs, skeleton, and part-to-object links
+ JSONL writer and optional annotated video writer

Usage:
  python run_pipeline.py \
    --source demo_16x9.mp4 \
    --yolo /path/to/yolo12s.pt \
    --pose_weight /path/to/yolov8n-pose.pt \
    --out out.jsonl \
    --save_vis vis_out.mp4 \
    --draw_parts \
    --show
"""

import os
import cv2
import argparse
from typing import Optional, Tuple, List, Dict

# Project-local imports
from src.detector_yolo import YoloDetector
from src.tracker import ByteTrackPair
from src.hake_adapter import HakeModel   # temporary heuristic; swap to real HAKE later
from src.pose_adapter import PoseEstimator
from src.utils_io import JsonlWriter
from src.schemas import FrameRecord, HOI

# ----------------------------
# Video helpers
# ----------------------------
def _open_video(source: str) -> cv2.VideoCapture:
    try:
        cam_idx = int(source)
        cap = cv2.VideoCapture(cam_idx)
    except ValueError:
        cap = cv2.VideoCapture(source)
    return cap

def _build_meta(source: str, cap: cv2.VideoCapture, yolo_path: str, hake_path: Optional[str]) -> dict:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return {
        "id": os.path.basename(str(source)) + "_session",
        "source": source,
        "fps": float(fps),
        "resolution": [W, H],
        "model": {
            "detector": os.path.basename(yolo_path),
            "hake": os.path.basename(hake_path) if hake_path else None,
            "tracker": "ByteTrack(supervision)"
        }
    }

def _start_writer(save_path: Optional[str], fps: float, size: Tuple[int, int]):
    if not save_path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    W, H = size
    return cv2.VideoWriter(save_path, fourcc, fps, (W, H))

# ----------------------------
# Drawing helpers (BGR colors)
# ----------------------------
COL_HUM = (0, 200, 0)       # green
COL_OBJ = (60, 160, 255)    # orange-ish
COL_TXT = (255, 255, 255)   # white
COL_REL = (255, 0, 255)     # magenta
COL_SKL = (0, 255, 255)     # yellow-cyan for skeleton
COL_PART = (0, 200, 255)    # link from part to object

COCO_EDGES = [
    # head/face
    (0,1), (0,2), (1,3), (2,4),
    # torso
    (5,6), (5,11), (6,12), (11,12),
    # left arm
    (5,7), (7,9),
    # right arm
    (6,8), (8,10),
    # left leg
    (11,13), (13,15),
    # right leg
    (12,14), (14,16)
]
LEFT_WRIST, RIGHT_WRIST = 9, 10


def _put_label(img, text, x, y, bg_col=(0, 0, 0), txt_col=COL_TXT, fs=0.5, th=1):
    (tw, th_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cv2.rectangle(img, (x, y - th_text - 4), (x + tw + 2, y), bg_col, -1)
    cv2.putText(img, text, (x + 1, y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, txt_col, th, cv2.LINE_AA)

def _center(b):
    x1, y1, x2, y2 = b
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def draw_overlay(
    frame,
    persons: List[Dict],
    objects: List[Dict],
    hoi_list: List[Dict],
    draw_parts: bool = False,
    font_scale: float = 0.6,
    thickness: int = 2
):
    # Humans
    for p in persons:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_HUM, thickness)
        hid = p.get("track_id", -1)
        _put_label(frame, f"H{hid}", x1, y1, bg_col=(0, 100, 0), fs=font_scale, th=max(1, thickness-1))

    # Objects
    for o in objects:
        x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_OBJ, thickness)
        oid = o.get("track_id", -1)
        cat = o.get("category", "obj")
        _put_label(frame, f"O{oid}:{cat}", x1, y1, bg_col=(60, 80, 140), fs=font_scale, th=max(1, thickness-1))

    # Skeleton (if keypoints exist)
    JOINT_RADIUS = 3
    CONF_TH = 0.2
    
    for p in persons:
        kps = p.get("keypoints")
        if not kps:
            continue
    
        # draw edges
        for (a, b) in COCO_EDGES:
            if a < len(kps) and b < len(kps):
                xa, ya, ca = kps[a]
                xb, yb, cb = kps[b]
                if ca > CONF_TH and cb > CONF_TH:
                    cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), COL_SKL, 2, cv2.LINE_AA)
    
        # draw all joints
        for j, (x, y, c) in enumerate(kps):
            if c > CONF_TH:
                cv2.circle(frame, (int(x), int(y)), JOINT_RADIUS, COL_SKL, -1, cv2.LINE_AA)


    # Relations (center-to-center)
    for h in hoi_list:
        hi = h["human_id"]
        oi = h["object_id"]
        verb = h["verb"]
        ph = next((p for p in persons if int(p.get("track_id", -1)) == int(hi)), None)
        po = next((o for o in objects if int(o.get("track_id", -1)) == int(oi)), None)
        if ph is None or po is None:
            continue
        ch = _center(ph["bbox"])
        co = _center(po["bbox"])
        cv2.line(frame, ch, co, COL_REL, max(1, thickness - 1), cv2.LINE_AA)
        mx, my = int((ch[0] + co[0]) / 2), int((ch[1] + co[1]) / 2)
        _put_label(frame, verb, mx, my, bg_col=(120, 0, 120), fs=font_scale, th=max(1, thickness-1))

    # Optional: part-to-object link for 'hold'
    if draw_parts:
        for h in hoi_list:
            if h["verb"] != "hold":
                continue
            hi = h["human_id"]; oi = h["object_id"]
            ph = next((p for p in persons if int(p.get("track_id", -1)) == int(hi)), None)
            po = next((o for o in objects if int(o.get("track_id", -1)) == int(oi)), None)
            if ph is None or po is None:
                continue

            kps = ph.get("keypoints")
            if not kps:
                continue

            wrists = []
            for wi, name in [(LEFT_WRIST, "left_hand"), (RIGHT_WRIST, "right_hand")]:
                if wi < len(kps) and kps[wi][2] > 0.2:
                    wrists.append(((int(kps[wi][0]), int(kps[wi][1])), name))
            if not wrists:
                continue

            oc = _center(po["bbox"])
            # choose wrist closest to object center
            hand_pt, hand_name = min(
                wrists, key=lambda t: (t[0][0]-oc[0])**2 + (t[0][1]-oc[1])**2
            )
            cv2.line(frame, hand_pt, oc, COL_PART, 2, cv2.LINE_AA)
            label = f"{hand_name}-hold-{po.get('category','obj')}"
            (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            px = int((hand_pt[0] + oc[0]) / 2 - tw / 2)
            py = int((hand_pt[1] + oc[1]) / 2)
            cv2.rectangle(frame, (px, py - th_text - 4), (px + tw + 2, py), (20, 80, 120), -1)
            cv2.putText(frame, label, (px + 1, py - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="YOLOv11/12 + ByteTrack + HAKE -> JSONL + visualization + part links")
    ap.add_argument("--source", required=True, help="Video file path or camera index (e.g., 0)")
    ap.add_argument("--yolo", required=True, help="YOLO weight (.pt) path")
    ap.add_argument("--out", default="output.jsonl", help="Output JSONL path")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence")
    ap.add_argument("--iou", type=float, default=0.5, help="YOLO IoU")
    ap.add_argument("--device", default=None, help="YOLO device (e.g., 'cuda:0' or 'cpu')")
    ap.add_argument("--hake_weight", default=None, help="Real HAKE weight (if you wire it)")
    # pose / overlay
    ap.add_argument("--pose_weight", default=None, help="Ultralytics pose weight (e.g., yolov8n-pose.pt)")
    ap.add_argument("--draw_parts", action="store_true", help="Draw skeleton and wrist-to-object links for 'hold'")
    ap.add_argument("--save_vis", default=None, help="Save annotated video to this path (e.g., vis_out.mp4)")
    ap.add_argument("--show", action="store_true", help="Display annotated frames in a window")
    ap.add_argument("--font_scale", type=float, default=0.6, help="Overlay text size")
    ap.add_argument("--thickness", type=int, default=2, help="Box/line thickness")
    args = ap.parse_args()

    cap = _open_video(args.source)
    assert cap.isOpened(), f"Failed to open source: {args.source}"

    detector = YoloDetector(args.yolo, conf=args.conf, iou=args.iou, device=args.device)
    hake = HakeModel(weight_path=args.hake_weight, device=args.device)  # heuristic until real model is wired
    tracker = ByteTrackPair(object_id_offset=100000)

    pose_est = PoseEstimator(args.pose_weight, conf=args.conf, iou=args.iou, device=args.device) if args.pose_weight else None

    meta = _build_meta(args.source, cap, args.yolo, args.hake_weight)
    writer = JsonlWriter(args.out, meta=meta)

    fps = meta["fps"]
    W, H = meta["resolution"]
    vis_writer = _start_writer(args.save_vis, fps=fps, size=(W, H)) if args.save_vis else None

    idx = 0
    win_name = "HOI-Overlay" if args.show else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Detection
        persons, objects = detector(frame)

        # 2) Tracking
        persons, objects = tracker.step(persons, objects)

        # 3) Pose (optional) -> attach keypoints to persons
        if pose_est is not None:
            persons = pose_est(frame, persons)

        # 4) HAKE (relations only)
        hoi_raw = hake.infer(frame, persons, objects)

        # 5) Build HOI list with tracked IDs
        hoi_list = []
        for h in hoi_raw:
            hi, oi = h["human_idx"], h["object_idx"]
            if hi < len(persons) and oi < len(objects):
                verb = h["verb"]; score = float(h["score"])
                part = h.get("part"); part_score = h.get("part_score", None)
                obj_cat = objects[oi].get("category", "object")
                hoi_list.append(
                    HOI(
                        human_id=int(persons[hi].get("track_id", -1)),
                        object_id=int(objects[oi].get("track_id", -1)),
                        verb=verb,
                        score=score,
                        part=part,
                        part_score=float(part_score) if part_score is not None else None,
                        triplet=["person", verb, obj_cat],
                    ).model_dump()
                )

        # 6) Visualization overlay (incl. skeleton and part links)
        draw_overlay(
            frame,
            persons=persons,
            objects=objects,
            hoi_list=hoi_list,
            draw_parts=args.draw_parts,
            font_scale=args.font_scale,
            thickness=args.thickness
        )

        # 7) JSONL write
        rec = FrameRecord(
            frame_index=idx,
            timestamp_ms=int((idx / max(1.0, fps)) * 1000),
            humans=[
                {"track_id": p.get("track_id"), "bbox_xyxy": p["bbox"], "score": float(p.get("score", 1.0))}
                for p in persons
            ],
            objects=[
                {
                    "track_id": o.get("track_id"),
                    "bbox_xyxy": o["bbox"],
                    "category": o.get("category", "object"),
                    "score": float(o.get("score", 1.0)),
                }
                for o in objects
            ],
            hoi=hoi_list,
        ).model_dump()
        writer.write_line(rec)

        # 8) Save / Show
        if vis_writer is not None:
            vis_writer.write(frame)
        if args.show:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        idx += 1

    writer.close()
    if vis_writer is not None:
        vis_writer.release()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Done. JSONL written to: {args.out}")
    if args.save_vis:
        print(f"Annotated video written to: {args.save_vis}")


if __name__ == "__main__":
    main()
