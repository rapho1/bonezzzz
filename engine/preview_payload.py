"""
Builds a compact skeleton-preview payload that the frontend renders in three.js.
Works from keypoint frames (MediaPipe world landmarks).

Payload shape:
    {
      "fps": float,
      "names": [joint names...],
      "bones": [[i, j], ...],         # index pairs into each frame's point list
      "frames": [ [[x,y,z], ...], ... ]   # subsampled, y-up, centimetres
    }
"""

# MediaPipe indices we keep for the stick figure.
KEY_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
NAMES = ["nose", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
         "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank"]

# Bones as index pairs into the reduced KEY_IDX list (positions 0..12).
BONES = [
    (1, 2), (1, 7), (2, 8), (7, 8),       # shoulders, torso sides, hips
    (1, 3), (3, 5), (2, 4), (4, 6),       # arms
    (7, 9), (9, 11), (8, 10), (10, 12),   # legs
    (0, 1), (0, 2),                       # head to shoulders
]

_SCALE = 100.0


def build_preview(frames: list, fps: float, max_frames: int = 90) -> dict:
    valid = [(i, f) for i, f in enumerate(frames) if f is not None]
    if not valid:
        return {"fps": fps, "names": NAMES, "bones": BONES, "frames": []}

    n = len(valid)
    step = max(1, n // max_frames)
    out_frames = []
    for k in range(0, n, step):
        _, f = valid[k]
        pts = []
        for idx in KEY_IDX:
            lm = f[idx]
            pts.append([lm["x"] * _SCALE, -lm["y"] * _SCALE, -lm["z"] * _SCALE])
        out_frames.append(pts)

    # We subsample by `step`, so the preview must be played at fps/step to match
    # the real-time speed of the source video (otherwise it runs `step`x too fast).
    return {"fps": fps / step, "names": NAMES, "bones": BONES, "frames": out_frames}
