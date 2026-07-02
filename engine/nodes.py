"""
Node compute functions. Each node takes its parameters and upstream results,
and returns a result dict tagged with a "kind".

Result kinds:
  video      -> {"kind":"video", "path", "fps", "count"}
  keypoints  -> {"kind":"keypoints", "fps", "frames"}      (frames may contain None)
  bvh        -> {"kind":"bvh", "fps", "bvh", "frames"}
"""
import json
import os

import cv2

from pipeline.smooth import smooth_landmarks
from pipeline.to_bvh import bvh_string
from engine.backends import run_backend

HEAVY_TYPES = {"pose_estimation"}
CACHE_DIR = "cache"


def _probe_video(path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, count


def run_video_input(params: dict, _parents: list) -> dict:
    source_type = params.get("source_type", "video")
    path = params.get("path", "")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Input path not found: {path!r}")

    if source_type == "keypoints":
        d = json.load(open(path))
        return {"kind": "keypoints", "fps": float(d["fps"]), "frames": d["frames"]}

    fps, count = _probe_video(path)
    return {"kind": "video", "path": path, "fps": fps, "count": count}


def run_pose_estimation(params: dict, parents: list, key: str) -> dict:
    src = parents[0]
    if src["kind"] == "keypoints":
        return src  # passthrough if upstream already provides keypoints

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"pose_{key}.json")
    if os.path.exists(cache_path):
        d = json.load(open(cache_path))
        res = {"kind": "keypoints", "fps": float(d["fps"]), "frames": d["frames"]}
        if d.get("bvh"):
            res["bvh"] = d["bvh"]  # WHAM's pre-baked angle BVH
        return res

    backend = params.get("backend", "mediapipe")
    frames, fps, bvh = run_backend(backend, src["path"], params)
    json.dump({"fps": fps, "frames": frames, "bvh": bvh}, open(cache_path, "w"))
    res = {"kind": "keypoints", "fps": fps, "frames": frames}
    if bvh:
        res["bvh"] = bvh
    return res


def run_smooth(params: dict, parents: list) -> dict:
    src = parents[0]
    sm = smooth_landmarks(
        src["frames"],
        src["fps"],
        cutoff_hz=float(params.get("cutoff", 6.0)),
        trim_edges=bool(params.get("trim_edges", True)),
    )
    res = {"kind": "keypoints", "fps": src["fps"], "frames": sm}
    # WHAM's angle BVH is already temporally smooth; carry it through untouched
    # (keypoint smoothing only affects the preview, not the exported angles).
    if src.get("bvh"):
        res["bvh"] = src["bvh"]
    return res


def run_output(params: dict, parents: list) -> dict:
    src = parents[0]
    frames = src["frames"]
    tpose = bool(params.get("tpose_start", False))
    # Prefer WHAM's direct SMPL-angle BVH; fall back to solving from landmarks.
    # (tpose_start only applies to the landmark solver - WHAM's BVH arrives
    # pre-baked from the WSL side.)
    txt = src.get("bvh") or bvh_string(frames, src["fps"], tpose_start=tpose)
    return {"kind": "bvh", "fps": src["fps"], "bvh": txt, "frames": frames}


def run_node(node_type: str, params: dict, parents: list, key: str) -> dict:
    if node_type == "video_input":
        return run_video_input(params, parents)
    if node_type == "pose_estimation":
        return run_pose_estimation(params, parents, key)
    if node_type == "smooth":
        return run_smooth(params, parents)
    if node_type == "output":
        return run_output(params, parents)
    raise ValueError(f"Unknown node type: {node_type}")


def node_info(result: dict) -> str:
    kind = result["kind"]
    if kind == "video":
        return f"{result['count']} frames @ {result['fps']:.0f} fps"
    if kind == "keypoints":
        detected = sum(1 for f in result["frames"] if f is not None)
        return f"{len(result['frames'])} frames ({detected} detected)"
    if kind == "bvh":
        return f"{len(result['frames'])} frames baked"
    return kind
