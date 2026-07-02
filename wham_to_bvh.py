"""
Runs inside the WSL `wham` conda env.

Converts WHAM's SMPL output DIRECTLY to BVH using the real per-joint SMPL
rotations (pose params are already local joint rotations relative to the parent),
instead of routing through 3D positions + a geometric solver. This preserves
WHAM's actual joint angles — smooth, plausible, no foot skating.

Also writes a keypoints JSON (SMPL joints -> our 13 landmarks) for the app's
live preview / graph flow.

Usage:
  python wham_to_bvh.py --video <path> --out_bvh <a.bvh> --out_json <b.json> \
                        --out_dir output/wham_cache
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# SMPL 24-joint kinematic tree
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
         "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
         "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
         "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"]
CHILDREN = {j: [c for c, p in enumerate(PARENTS) if p == j] for j in range(24)}
LEAVES = [j for j in range(24) if not CHILDREN[j]]
SCALE = 100.0  # metres -> centimetres (matches Blender import global_scale=0.01)

# SMPL joint -> MediaPipe landmark index (for the preview keypoints)
SMPL2MP = {15: 0, 16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16,
           1: 23, 2: 24, 4: 25, 5: 26, 7: 27, 8: 28}

# camera frame (y down, z forward) -> BVH y-up world (rotate 180 deg about X)
R_FLIP = Rotation.from_euler("x", 180, degrees=True).as_matrix()


def load_wham(video, out_dir):
    stem = os.path.splitext(os.path.basename(video))[0]
    pkl = os.path.join(out_dir, stem, "wham_output.pkl")
    if not os.path.exists(pkl):
        subprocess.run([sys.executable, "demo.py", "--video", video,
                        "--output_pth", out_dir, "--save_pkl", "--estimate_local_only"],
                       cwd="/root/WHAM", check=True)
    import joblib
    return joblib.load(pkl)


def smpl_model(betas_mean):
    import torch, smplx
    model = smplx.create("/root/WHAM/dataset/body_models", model_type="smpl",
                         gender="neutral", batch_size=1)
    with torch.no_grad():
        out = model(betas=torch.tensor(betas_mean[None], dtype=torch.float32))
    J = out.joints.numpy()[0, :24, :]  # rest-pose joint locations (metres)
    return model, J


def posed_joints(model, track):
    import torch
    N = track["pose"].shape[0]
    import smplx
    m = smplx.create("/root/WHAM/dataset/body_models", model_type="smpl",
                     gender="neutral", batch_size=N)
    pose = torch.tensor(track["pose"], dtype=torch.float32)
    betas = torch.tensor(track["betas"], dtype=torch.float32)
    trans = torch.tensor(track["trans"], dtype=torch.float32)
    with torch.no_grad():
        out = m(betas=betas, body_pose=pose[:, 3:], global_orient=pose[:, :3], transl=trans)
    return out.joints.numpy()[:, :24, :]


def build_hierarchy(offsets):
    """Return (hierarchy_text, dfs_order) for the SMPL skeleton."""
    lines = ["HIERARCHY", "ROOT Pelvis", "{"]
    dfs = []

    def off_str(j):
        o = offsets[j]
        return f"{o[0]:.4f} {o[1]:.4f} {o[2]:.4f}"

    def end_site(j, indent):
        pad = "\t" * indent
        o = offsets[j] * 0.5
        if np.linalg.norm(o) < 1e-3:
            o = np.array([0.0, 5.0, 0.0])
        lines.append(f"{pad}End Site")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}\tOFFSET {o[0]:.4f} {o[1]:.4f} {o[2]:.4f}")
        lines.append(f"{pad}}}")

    def recurse(j, indent):
        dfs.append(j)
        pad = "\t" * indent
        if j == 0:
            lines.append(f"{pad}\tOFFSET 0.00 0.00 0.00")
            lines.append(f"{pad}\tCHANNELS 6 Xposition Yposition Zposition "
                         "Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{pad}OFFSET {off_str(j)}")
            lines.append(f"{pad}CHANNELS 3 Zrotation Xrotation Yrotation")
        for c in CHILDREN[j]:
            lines.append(f"{pad}JOINT {NAMES[c]}")
            lines.append(f"{pad}{{")
            recurse(c, indent + 1)
            lines.append(f"{pad}}}")
        if not CHILDREN[j]:
            end_site(j, indent + 1)

    recurse(0, 0)
    lines.append("}")
    return "\n".join(lines) + "\n", dfs


def _wrap180(deg):
    """Wrap an angle (degrees) into (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def _decompose_zxy_continuous(Rj, prev):
    """
    scipy's as_euler("ZXY") decomposition isn't unique: (z, x, y) and
    (z+180, 180-x, y+180) represent the exact same rotation. Decomposing each
    frame independently picks a branch arbitrarily, which makes the BVH
    visibly "flip" a joint 180 degrees between two consecutive frames
    whenever the solver lands on a different branch than the previous frame
    (each frame's pose is individually correct, but the jump between frames
    reads as a snap during playback). Pick whichever branch stays closest to
    the previous frame's angles for this joint.
    """
    z, x, y = Rotation.from_matrix(Rj).as_euler("ZXY", degrees=True)
    if prev is None:
        return z, x, y
    alt = (_wrap180(z + 180.0), 180.0 - x, _wrap180(y + 180.0))
    primary = (z, x, y)
    d_primary = sum(_wrap180(primary[k] - prev[k]) ** 2 for k in range(3))
    d_alt = sum(_wrap180(alt[k] - prev[k]) ** 2 for k in range(3))
    return primary if d_primary <= d_alt else alt


def make_bvh(track, offsets, dfs, fps):
    pose = track["pose"]              # (N, 72) axis-angle
    trans = track["trans"]           # (N, 3)
    N = pose.shape[0]
    hierarchy, _ = build_hierarchy(offsets)

    lines = [hierarchy, "MOTION", f"Frames: {N}", f"Frame Time: {1.0 / fps:.6f}"]
    prev_angles = {}  # joint index -> (z, x, y) from the previous frame
    for i in range(N):
        vals = []
        # root translation (flip camera -> y-up) in cm
        t = R_FLIP @ trans[i]
        vals += [t[0] * SCALE, t[1] * SCALE, t[2] * SCALE]
        for j in dfs:
            aa = pose[i, 3 * j:3 * j + 3]
            Rj = Rotation.from_rotvec(aa).as_matrix()
            if j == 0:
                Rj = R_FLIP @ Rj  # orient whole body into y-up world
            z, x, y = _decompose_zxy_continuous(Rj, prev_angles.get(j))
            prev_angles[j] = (z, x, y)
            vals += [z, x, y]
        lines.append(" ".join(f"{v:.4f}" for v in vals))
    return "\n".join(lines) + "\n"


def make_keypoints(J_posed, fps):
    frames = []
    for i in range(J_posed.shape[0]):
        lms = [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0} for _ in range(33)]
        for smpl_i, mp_i in SMPL2MP.items():
            p = J_posed[i, smpl_i]
            lms[mp_i] = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                         "visibility": 1.0}
        frames.append(lms)
    return {"fps": fps, "frames": frames}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_bvh", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_dir", default="output/wham_cache")
    a = ap.parse_args()

    import cv2
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    data = load_wham(a.video, a.out_dir)
    tid = max(data.keys(), key=lambda k: data[k]["pose"].shape[0])
    track = data[tid]

    betas_mean = track["betas"].mean(axis=0)
    model, J_rest = smpl_model(betas_mean)
    offsets = np.zeros((24, 3))
    for j in range(1, 24):
        offsets[j] = (J_rest[j] - J_rest[PARENTS[j]]) * SCALE

    _, dfs = build_hierarchy(offsets)
    bvh = make_bvh(track, offsets, dfs, fps)
    with open(a.out_bvh, "w") as f:
        f.write(bvh)

    Jp = posed_joints(model, track)
    # Leave keypoints in raw SMPL/camera convention (y-down); build_preview
    # applies the same [x,-y,-z] flip it uses for MediaPipe, giving an upright
    # viewport. (The exported BVH handles its own orientation via R_FLIP.)
    kp = make_keypoints(Jp, fps)
    with open(a.out_json, "w") as f:
        json.dump(kp, f)

    print(f"WROTE bvh={a.out_bvh} json={a.out_json} frames={track['pose'].shape[0]} track={tid}")
