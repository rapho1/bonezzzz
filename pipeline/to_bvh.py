"""
Converts smoothed MediaPipe world landmarks to BVH format.

KEY IDEA (this is what makes the animation actually work):
BVH joint rotations are HIERARCHICAL — each joint's channels store a rotation
relative to its PARENT's coordinate frame. So we:
  1. Compute each joint's WORLD rotation from the 3D keypoints.
  2. Localize it:  R_local[j] = R_world[parent]^T @ R_world[j]
  3. Decompose R_local into Euler ZXY angles for the channels.
Under forward kinematics the product of locals telescopes back to the world
rotation, so every bone points where the keypoints say it should — no tearing,
no flipping.

MediaPipe Pose landmark indices used:
  0 nose | 11/12 shoulders | 13/14 elbows | 15/16 wrists
  23/24 hips | 25/26 knees | 27/28 ankles
"""
import math
import numpy as np


LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

# Skeleton, depth-first order. Channels are written in exactly this order.
JOINT_ORDER = [
    "Hips", "Spine", "Neck", "Head",
    "LeftArm", "LeftForeArm", "LeftHand",
    "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot",
]

PARENT = {
    "Hips": None, "Spine": "Hips", "Neck": "Spine", "Head": "Neck",
    "LeftArm": "Spine", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "RightArm": "Spine", "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg",
    "RightUpLeg": "Hips", "RightLeg": "RightUpLeg", "RightFoot": "RightLeg",
}

# Leaf joints carry zero rotation (endpoints).
LEAVES = {"Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot"}

_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])


def _lm(frame: list[dict], name: str) -> np.ndarray:
    """MediaPipe world landmark -> standard y-up, z-forward(toward camera)."""
    l = frame[LM[name]]
    return np.array([l["x"], -l["y"], -l["z"]], dtype=float)


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector a to unit vector b (swing)."""
    a = _norm(a)
    b = _norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 180 deg: rotate around any axis perpendicular to a
        axis = _norm(np.cross(a, _X if abs(a[0]) < 0.9 else _Y))
        return _axis_angle(axis, math.pi)
    return _axis_angle(v / s, math.atan2(s, c))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1 - c
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
    ])


def _basis(up: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Full orthonormal world rotation from a rest frame (right=+X, up=+Y, fwd=+Z)
    to the current frame defined by `up` and `right`. Captures torso yaw/turn.
    Returned matrix has the new axes as COLUMNS.
    """
    u = _norm(up)
    f = _norm(np.cross(_norm(right), u))
    r = _norm(np.cross(u, f))
    if np.linalg.norm(f) < 1e-6:
        return np.eye(3)
    return np.column_stack([r, u, f])


def _world_rotations(frame: list[dict]) -> dict[str, np.ndarray]:
    nose = _lm(frame, "nose")
    lsho, rsho = _lm(frame, "left_shoulder"), _lm(frame, "right_shoulder")
    lel, rel = _lm(frame, "left_elbow"), _lm(frame, "right_elbow")
    lwr, rwr = _lm(frame, "left_wrist"), _lm(frame, "right_wrist")
    lhip, rhip = _lm(frame, "left_hip"), _lm(frame, "right_hip")
    lkn, rkn = _lm(frame, "left_knee"), _lm(frame, "right_knee")
    lank, rank = _lm(frame, "left_ankle"), _lm(frame, "right_ankle")

    mid_hip = (lhip + rhip) * 0.5
    mid_sho = (lsho + rsho) * 0.5

    R = {}
    # Torso joints use a full basis so body turning is captured.
    R["Hips"] = _basis(up=mid_sho - mid_hip, right=rhip - lhip)
    R["Spine"] = _basis(up=nose - mid_sho, right=rsho - lsho)
    R["Neck"] = _rot_between(_Y, nose - mid_sho)
    # Limbs: swing from rest direction to current bone direction.
    R["LeftArm"] = _rot_between(-_X, lel - lsho)
    R["LeftForeArm"] = _rot_between(-_X, lwr - lel)
    R["RightArm"] = _rot_between(_X, rel - rsho)
    R["RightForeArm"] = _rot_between(_X, rwr - rel)
    R["LeftUpLeg"] = _rot_between(-_Y, lkn - lhip)
    R["LeftLeg"] = _rot_between(-_Y, lank - lkn)
    R["RightUpLeg"] = _rot_between(-_Y, rkn - rhip)
    R["RightLeg"] = _rot_between(-_Y, rank - rkn)
    # Leaves.
    for leaf in LEAVES:
        R[leaf] = np.eye(3)

    return R, mid_hip


def _decompose_zxy(R: np.ndarray) -> tuple[float, float, float]:
    """
    Decompose rotation matrix R = Rz @ Rx @ Ry into (z, x, y) degrees,
    matching BVH channel order 'Zrotation Xrotation Yrotation'.
    """
    sx = max(-1.0, min(1.0, R[2, 1]))
    x = math.asin(sx)
    if abs(math.cos(x)) > 1e-6:
        z = math.atan2(-R[0, 1], R[1, 1])
        y = math.atan2(-R[2, 0], R[2, 2])
    else:  # gimbal lock
        z = math.atan2(R[1, 0], R[0, 0])
        y = 0.0
    return math.degrees(z), math.degrees(x), math.degrees(y)


def _frame_channels(frame: list[dict], scale: float = 100.0) -> list[float]:
    Rworld, mid_hip = _world_rotations(frame)

    channels: list[float] = []
    for joint in JOINT_ORDER:
        parent = PARENT[joint]
        if parent is None:
            R_local = Rworld[joint]
        else:
            R_local = Rworld[parent].T @ Rworld[joint]
        z, x, y = _decompose_zxy(R_local)

        if joint == "Hips":
            pos = mid_hip * scale
            channels.extend([pos[0], pos[1], pos[2], z, x, y])
        else:
            channels.extend([z, x, y])

    return channels


# ---------------------------------------------------------------------------
# BVH hierarchy. Offsets only set rest proportions/directions; angles drive it.
# ---------------------------------------------------------------------------
_HIERARCHY = """\
HIERARCHY
ROOT Hips
{
\tOFFSET 0.00 0.00 0.00
\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
\tJOINT Spine
\t{
\t\tOFFSET 0.00 50.00 0.00
\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\tJOINT Neck
\t\t{
\t\t\tOFFSET 0.00 20.00 0.00
\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\tJOINT Head
\t\t\t{
\t\t\t\tOFFSET 0.00 10.00 0.00
\t\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\t\tEnd Site
\t\t\t\t{
\t\t\t\t\tOFFSET 0.00 10.00 0.00
\t\t\t\t}
\t\t\t}
\t\t}
\t\tJOINT LeftArm
\t\t{
\t\t\tOFFSET -18.00 0.00 0.00
\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\tJOINT LeftForeArm
\t\t\t{
\t\t\t\tOFFSET -28.00 0.00 0.00
\t\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\t\tJOINT LeftHand
\t\t\t\t{
\t\t\t\t\tOFFSET -25.00 0.00 0.00
\t\t\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\t\t\tEnd Site
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET -8.00 0.00 0.00
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t\tJOINT RightArm
\t\t{
\t\t\tOFFSET 18.00 0.00 0.00
\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\tJOINT RightForeArm
\t\t\t{
\t\t\t\tOFFSET 28.00 0.00 0.00
\t\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\t\tJOINT RightHand
\t\t\t\t{
\t\t\t\t\tOFFSET 25.00 0.00 0.00
\t\t\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\t\t\tEnd Site
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET 8.00 0.00 0.00
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t}
\tJOINT LeftUpLeg
\t{
\t\tOFFSET -10.00 0.00 0.00
\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\tJOINT LeftLeg
\t\t{
\t\t\tOFFSET 0.00 -45.00 0.00
\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\tJOINT LeftFoot
\t\t\t{
\t\t\t\tOFFSET 0.00 -45.00 0.00
\t\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\t\tEnd Site
\t\t\t\t{
\t\t\t\t\tOFFSET 0.00 -5.00 12.00
\t\t\t\t}
\t\t\t}
\t\t}
\t}
\tJOINT RightUpLeg
\t{
\t\tOFFSET 10.00 0.00 0.00
\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\tJOINT RightLeg
\t\t{
\t\t\tOFFSET 0.00 -45.00 0.00
\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\tJOINT RightFoot
\t\t\t{
\t\t\t\tOFFSET 0.00 -45.00 0.00
\t\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\t\tEnd Site
\t\t\t\t{
\t\t\t\t\tOFFSET 0.00 -5.00 12.00
\t\t\t\t}
\t\t\t}
\t\t}
\t}
}
"""


def bvh_string(smoothed_frames: list[list[dict]], fps: float) -> str:
    """Return the full BVH file contents as a string."""
    frame_time = 1.0 / fps
    n_frames = len(smoothed_frames)

    lines = [_HIERARCHY, "MOTION\n", f"Frames: {n_frames}\n",
             f"Frame Time: {frame_time:.6f}\n"]
    for frame in smoothed_frames:
        ch = _frame_channels(frame)
        lines.append(" ".join(f"{v:.4f}" for v in ch) + "\n")
    return "".join(lines)


def generate_bvh(
    smoothed_frames: list[list[dict]],
    fps: float,
    output_path: str,
) -> None:
    text = bvh_string(smoothed_frames, fps)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"Saved BVH -> {output_path}  ({len(smoothed_frames)} frames @ {fps:.1f} fps)")
