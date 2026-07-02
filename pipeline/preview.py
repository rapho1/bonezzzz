"""
Standalone BVH previewer: parses a BVH, runs forward kinematics, and renders
either a montage PNG of sampled frames or an animated GIF. Lets you see the
skeleton without opening Blender.

Usage:
    python pipeline/preview.py output/walk.bvh                 # montage PNG
    python pipeline/preview.py output/walk.bvh --gif           # animated GIF
"""
import math
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_bvh(path: str):
    with open(path) as f:
        text = f.read()
    head, motion = text.split("MOTION")

    # --- parse hierarchy ---
    joints = []           # list of dicts: name, parent, offset, channels(list)
    stack = []
    name = None
    for line in head.splitlines():
        s = line.strip()
        if s.startswith("ROOT") or s.startswith("JOINT"):
            name = s.split()[1]
            parent = stack[-1] if stack else None
            joints.append({"name": name, "parent": parent,
                           "offset": np.zeros(3), "channels": []})
        elif s.startswith("End Site"):
            name = "__end__"
        elif s.startswith("OFFSET"):
            vals = [float(x) for x in s.split()[1:]]
            if name == "__end__":
                continue
            joints[-1]["offset"] = np.array(vals)
        elif s.startswith("CHANNELS"):
            parts = s.split()
            joints[-1]["channels"] = parts[2:]
        elif s == "{":
            stack.append(name)
        elif s == "}":
            if stack:
                stack.pop()

    index = {j["name"]: i for i, j in enumerate(joints)}

    # --- parse motion ---
    mlines = [l for l in motion.splitlines() if l.strip()]
    n_frames = int(mlines[0].split(":")[1])
    frame_time = float(mlines[1].split(":")[1])
    data = np.array([[float(v) for v in l.split()] for l in mlines[2:]])
    return joints, index, data, n_frames, frame_time


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk_positions(joints, frame_values):
    """Forward kinematics -> dict name -> world position (3,)."""
    pos = {}
    rot = {}
    cursor = 0
    for j in joints:
        local_t = np.zeros(3)
        R = np.eye(3)
        for ch in j["channels"]:
            v = frame_values[cursor]; cursor += 1
            if ch.endswith("position"):
                axis = ch[0]
                local_t["XYZ".index(axis)] = v
            else:
                R = R @ _rot(ch[0], v)
        parent = j["parent"]
        if parent is None:
            world_R = R
            world_p = j["offset"] + local_t
        else:
            world_R = rot[parent] @ R
            world_p = pos[parent] + rot[parent] @ j["offset"]
        rot[j["name"]] = world_R
        pos[j["name"]] = world_p
    return pos


def bones(joints):
    return [(j["parent"], j["name"]) for j in joints if j["parent"]]


def render_montage(path, out_png, n=6):
    joints, index, data, n_frames, ft = parse_bvh(path)
    bone_list = bones(joints)
    idxs = np.linspace(0, n_frames - 1, n).astype(int)

    fig = plt.figure(figsize=(3 * n, 4))
    for k, fi in enumerate(idxs):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        pos = fk_positions(joints, data[fi])
        for p, c in bone_list:
            a, b = pos[p], pos[c]
            ax.plot([a[0], b[0]], [a[2], b[2]], [a[1], b[1]], "-o",
                    color="#1f77b4", markersize=2, linewidth=1.5)
        ax.set_title(f"frame {fi}", fontsize=8)
        ax.set_box_aspect([1, 1, 1.6])
        ax.set_xlim(-80, 80); ax.set_ylim(-80, 80); ax.set_zlim(-100, 100)
        ax.view_init(elev=10, azim=-90)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle(f"{path}  ({n_frames} frames)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    print(f"Saved montage -> {out_png}")


def render_gif(path, out_gif, stride=3, max_frames=120):
    import matplotlib.animation as animation
    joints, index, data, n_frames, ft = parse_bvh(path)
    bone_list = bones(joints)
    frames = list(range(0, n_frames, stride))[:max_frames]

    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111, projection="3d")

    def draw(fi):
        ax.cla()
        pos = fk_positions(joints, data[fi])
        for p, c in bone_list:
            a, b = pos[p], pos[c]
            ax.plot([a[0], b[0]], [a[2], b[2]], [a[1], b[1]], "-o",
                    color="#1f77b4", markersize=3, linewidth=2)
        ax.set_box_aspect([1, 1, 1.6])
        ax.set_xlim(-80, 80); ax.set_ylim(-80, 80); ax.set_zlim(-100, 100)
        ax.view_init(elev=10, azim=-90)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"frame {fi}")

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=80)
    anim.save(out_gif, writer="pillow", fps=12)
    print(f"Saved GIF -> {out_gif}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/preview.py <file.bvh> [--gif]")
        sys.exit(1)
    bvh = sys.argv[1]
    if "--gif" in sys.argv:
        render_gif(bvh, bvh.replace(".bvh", "_preview.gif"))
    else:
        render_montage(bvh, bvh.replace(".bvh", "_preview.png"))
