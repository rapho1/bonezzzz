"""
Forward-kinematics self-check for the BVH exporter.

Rebuilds each joint's WORLD rotation from the exported Euler channels using the
same hierarchy + composition convention a viewer uses (R = Rz @ Rx @ Ry,
world = parent_world @ local), then checks that each bone points in the same
direction as the original MediaPipe keypoints. If the max angular error is tiny,
the export math (localization + decomposition) is correct and the animation
will not tear or flip.
"""
import math
import numpy as np

from pipeline.to_bvh import (
    JOINT_ORDER, PARENT, LEAVES, _world_rotations, _decompose_zxy,
    _frame_channels, _axis_angle, _X, _Y,
)

# Rest direction of the bone each joint controls (for swing joints).
REST_DIR = {
    "Neck": _Y,
    "LeftArm": -_X, "LeftForeArm": -_X,
    "RightArm": _X, "RightForeArm": _X,
    "LeftUpLeg": -_Y, "LeftLeg": -_Y,
    "RightUpLeg": -_Y, "RightLeg": -_Y,
}

# How to get the actual world bone direction from landmarks (parent, child).
from pipeline.to_bvh import _lm
def _bone_dirs(frame):
    nose = _lm(frame, "nose")
    lsho, rsho = _lm(frame, "left_shoulder"), _lm(frame, "right_shoulder")
    lel, rel = _lm(frame, "left_elbow"), _lm(frame, "right_elbow")
    lwr, rwr = _lm(frame, "left_wrist"), _lm(frame, "right_wrist")
    lhip, rhip = _lm(frame, "left_hip"), _lm(frame, "right_hip")
    lkn, rkn = _lm(frame, "left_knee"), _lm(frame, "right_knee")
    lank, rank = _lm(frame, "left_ankle"), _lm(frame, "right_ankle")
    mid_sho = (lsho + rsho) * 0.5
    return {
        "Neck": nose - mid_sho,
        "LeftArm": lel - lsho, "LeftForeArm": lwr - lel,
        "RightArm": rel - rsho, "RightForeArm": rwr - rel,
        "LeftUpLeg": lkn - lhip, "LeftLeg": lank - lkn,
        "RightUpLeg": rkn - rhip, "RightLeg": rank - rkn,
    }


def _rot_z(d): c,s=math.cos(d),math.sin(d); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def _rot_x(d): c,s=math.cos(d),math.sin(d); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def _rot_y(d): c,s=math.cos(d),math.sin(d); return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def _recompose(z, x, y):
    return _rot_z(math.radians(z)) @ _rot_x(math.radians(x)) @ _rot_y(math.radians(y))


def verify_frame(frame) -> float:
    """Returns max angular error (degrees) over all swing bones for one frame."""
    # Export -> channels
    ch = _frame_channels(frame)
    # Re-parse channels back into per-joint (z,x,y)
    locals_zxy = {}
    idx = 0
    for joint in JOINT_ORDER:
        if joint == "Hips":
            idx += 3  # skip position
        z, x, y = ch[idx], ch[idx+1], ch[idx+2]
        locals_zxy[joint] = (z, x, y)
        idx += 3

    # FK: world rotation per joint from the LOCAL channels.
    Rfk = {}
    for joint in JOINT_ORDER:
        Rl = _recompose(*locals_zxy[joint])
        parent = PARENT[joint]
        Rfk[joint] = Rl if parent is None else Rfk[parent] @ Rl

    actual = _bone_dirs(frame)
    max_err = 0.0
    for joint, rest in REST_DIR.items():
        recon = Rfk[joint] @ rest
        recon = recon / (np.linalg.norm(recon) + 1e-9)
        want = actual[joint] / (np.linalg.norm(actual[joint]) + 1e-9)
        dot = max(-1.0, min(1.0, float(np.dot(recon, want))))
        err = math.degrees(math.acos(dot))
        max_err = max(max_err, err)
    return max_err


if __name__ == "__main__":
    import sys, json
    from pipeline.extract_pose import extract_pose
    from pipeline.smooth import smooth_landmarks

    video = sys.argv[1] if len(sys.argv) > 1 else "test_videos/walk.mp4"
    frames, fps = extract_pose(video)
    sm = smooth_landmarks(frames, fps)

    errs = [verify_frame(f) for f in sm]
    errs = np.array(errs)
    print(f"\nFK self-check over {len(errs)} frames:")
    print(f"  mean bone-direction error: {errs.mean():.4f} deg")
    print(f"  max  bone-direction error: {errs.max():.4f} deg")
    if errs.max() < 0.5:
        print("  PASS - export math is correct (bones reconstruct exactly).")
    else:
        print("  FAIL - rotations do not reconstruct input. Bug remains.")
