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


def _continuous_swing(
    rest: np.ndarray, target: np.ndarray,
    prev: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Minimal rotation taking `rest` to `target`, kept continuous across frames.

    Computing this fresh from the fixed rest direction every frame is
    numerically unstable whenever `target` swings near the antipodal point
    (180 deg from rest) - the rotation axis there is cross(rest, target)
    normalized, and near that point a tiny change in `target` swings that
    axis wildly (it's genuinely undefined exactly at 180 deg), producing a
    visibly erratic bone flip even though the keypoints move smoothly.
    Instead, once we have a previous frame, we rotate incrementally: minimal
    rotation from the PREVIOUS target direction to the CURRENT one, composed
    onto the previous world rotation. Consecutive video frames move by a
    small angle, so this almost never approaches the singularity (only the
    very first frame, or a frame right after a detection gap, uses the
    fixed-rest-direction formula directly).
    """
    target = _norm(target)
    if prev is None:
        R = _rot_between(rest, target)
    else:
        prev_target, prev_R = prev
        R = _rot_between(prev_target, target) @ prev_R
    return R, (target, R)


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


def _world_rotations(
    frame: list[dict],
    prev_swing: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    nose = _lm(frame, "nose")
    lsho, rsho = _lm(frame, "left_shoulder"), _lm(frame, "right_shoulder")
    lel, rel = _lm(frame, "left_elbow"), _lm(frame, "right_elbow")
    lwr, rwr = _lm(frame, "left_wrist"), _lm(frame, "right_wrist")
    lhip, rhip = _lm(frame, "left_hip"), _lm(frame, "right_hip")
    lkn, rkn = _lm(frame, "left_knee"), _lm(frame, "right_knee")
    lank, rank = _lm(frame, "left_ankle"), _lm(frame, "right_ankle")

    mid_hip = (lhip + rhip) * 0.5
    mid_sho = (lsho + rsho) * 0.5

    prev_swing = prev_swing or {}
    new_swing: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def swing(name: str, rest: np.ndarray, target: np.ndarray) -> np.ndarray:
        Rj, state = _continuous_swing(rest, target, prev_swing.get(name))
        new_swing[name] = state
        return Rj

    R = {}
    # Torso joints use a full basis so body turning is captured.
    R["Hips"] = _basis(up=mid_sho - mid_hip, right=rhip - lhip)
    R["Spine"] = _basis(up=nose - mid_sho, right=rsho - lsho)
    R["Neck"] = swing("Neck", _Y, nose - mid_sho)
    # Limbs: swing from rest direction to current bone direction.
    R["LeftArm"] = swing("LeftArm", -_X, lel - lsho)
    R["LeftForeArm"] = swing("LeftForeArm", -_X, lwr - lel)
    R["RightArm"] = swing("RightArm", _X, rel - rsho)
    R["RightForeArm"] = swing("RightForeArm", _X, rwr - rel)
    R["LeftUpLeg"] = swing("LeftUpLeg", -_Y, lkn - lhip)
    R["LeftLeg"] = swing("LeftLeg", -_Y, lank - lkn)
    R["RightUpLeg"] = swing("RightUpLeg", -_Y, rkn - rhip)
    R["RightLeg"] = swing("RightLeg", -_Y, rank - rkn)
    # Leaves.
    for leaf in LEAVES:
        R[leaf] = np.eye(3)

    return R, mid_hip, new_swing


def _decompose_zxy(R: np.ndarray, prev: tuple[float, float, float] | None = None
                   ) -> tuple[float, float, float]:
    """
    Decompose rotation matrix R = Rz @ Rx @ Ry into (z, x, y) degrees,
    matching BVH channel order 'Zrotation Xrotation Yrotation'.

    A ZXY decomposition of a given rotation is not unique: (z, x, y) and
    (z+180, 180-x, y+180) represent the exact same rotation. Decomposing each
    frame independently picks between these two branches arbitrarily, which
    is fine for a single static pose but makes an *animated* BVH visibly
    "flip" a joint 180 degrees between two consecutive frames whenever the
    solver happens to land on different branches for each (bones are
    correct at every individual frame, but the jump between frames reads as
    a snap/flip during playback). When `prev` (the previous frame's angles
    for this same joint) is given, we pick whichever branch is closer to it,
    keeping the animation continuous.
    """
    sx = max(-1.0, min(1.0, R[2, 1]))
    x = math.asin(sx)
    if abs(math.cos(x)) > 1e-6:
        z = math.atan2(-R[0, 1], R[1, 1])
        y = math.atan2(-R[2, 0], R[2, 2])
    else:  # gimbal lock
        z = math.atan2(R[1, 0], R[0, 0])
        y = 0.0
    z, x, y = math.degrees(z), math.degrees(x), math.degrees(y)

    if prev is None:
        return z, x, y

    alt = (_wrap180(z + 180.0), 180.0 - x, _wrap180(y + 180.0))
    primary = (z, x, y)

    def dist(a):
        return sum(_wrap180(a[i] - prev[i]) ** 2 for i in range(3))

    return primary if dist(primary) <= dist(alt) else alt


def _wrap180(deg: float) -> float:
    """Wrap an angle (degrees) into (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def _frame_channels(
    frame: list[dict],
    scale: float = 100.0,
    prev_angles: dict[str, tuple[float, float, float]] | None = None,
    prev_swing: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[float] | tuple[list[float], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    prev_angles: joint name -> (z, x, y) from the PREVIOUS frame, used to pick
    the continuity-preserving branch of the ZXY decomposition (see
    _decompose_zxy). Mutated in place with this frame's angles so the caller
    can pass it straight into the next frame's call.
    prev_swing: joint name -> (prev_target_dir, prev_world_R) from the
    PREVIOUS frame, used by _continuous_swing to avoid the antipodal
    instability in _rot_between (see _continuous_swing). When given, the
    return value becomes (channels, new_swing) so the caller can thread it
    into the next frame's call; when None, only channels is returned (used
    by verify_fk.py's single-frame self-consistency check, which has no
    notion of "previous frame").
    """
    Rworld, mid_hip, new_swing = _world_rotations(frame, prev_swing)

    channels: list[float] = []
    for joint in JOINT_ORDER:
        if joint in LEAVES:
            # MediaPipe gives us no hand/foot/head-twist orientation data, so
            # leaves carry no local rotation. (Rworld[leaf] is a fixed world
            # identity placeholder for the parent's decomposition elsewhere —
            # localizing THAT against a moving parent would produce the
            # parent's inverse, an erratic, gimbal-lock-prone value, not the
            # zero this joint is meant to be.)
            z, x, y = 0.0, 0.0, 0.0
            if prev_angles is not None:
                prev_angles[joint] = (z, x, y)
        else:
            parent = PARENT[joint]
            if parent is None:
                R_local = Rworld[joint]
            else:
                R_local = Rworld[parent].T @ Rworld[joint]
            prev = prev_angles.get(joint) if prev_angles is not None else None
            z, x, y = _decompose_zxy(R_local, prev)
            if prev_angles is not None:
                prev_angles[joint] = (z, x, y)

        if joint == "Hips":
            pos = mid_hip * scale
            channels.extend([pos[0], pos[1], pos[2], z, x, y])
        else:
            channels.extend([z, x, y])

    if prev_swing is None:
        return channels
    return channels, new_swing

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


def bvh_string(smoothed_frames: list[list[dict]], fps: float,
               tpose_start: bool = False) -> str:
    """
    Return the full BVH file contents as a string.

    tpose_start: prepend one frame with all rotations zeroed (the skeleton's
    rest T-pose, held at the first frame's root position). Riggers bind the
    character in a T-pose; providing it as a real baked frame means they
    don't have to overwrite frame 1 by hand.
    """
    frame_time = 1.0 / fps
    n_frames = len(smoothed_frames) + (1 if tpose_start else 0)

    lines = [_HIERARCHY, "MOTION\n", f"Frames: {n_frames}\n",
             f"Frame Time: {frame_time:.6f}\n"]
    # Seed the Euler-branch selection with zeros so the FIRST frame picks the
    # representation numerically closest to the rest T-pose (0,0,0). Without
    # this, frame 0 can legally come out as e.g. x=222 deg - the same physical
    # rotation as x=-138, but if the user keyframes a T-pose (all zeros)
    # right before it, Blender interpolates the raw channel value the long
    # way around, which reads as a 180-degree bone flip on playback.
    prev_angles: dict[str, tuple[float, float, float]] = {
        j: (0.0, 0.0, 0.0) for j in JOINT_ORDER}
    prev_swing: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    first = True
    for frame in smoothed_frames:
        ch, prev_swing = _frame_channels(
            frame, prev_angles=prev_angles, prev_swing=prev_swing)
        if first and tpose_start:
            # T-pose frame: fully neutral rest pose - ALL rotations zeroed,
            # including the root. An earlier version kept the root rotation
            # matching the actor's facing direction to avoid a jump into
            # frame 2, but that meant the "T-pose" could face sideways or
            # backward (whatever direction the actor faced in the source
            # footage) - useless for binding, since riggers need a pose that
            # actually looks like a neutral T-pose. Only root POSITION is
            # kept (so the rig doesn't teleport); the pose-to-frame-2 jump is
            # a single, instant, un-interpolated keyframe step (exactly what
            # manually resetting frame 1 by hand already did) - not a
            # multi-frame "spin", since BVH import bakes one keyframe per
            # source frame with no in-betweens.
            tpose = list(ch)
            for i in range(3, len(tpose)):
                tpose[i] = 0.0
            lines.append(" ".join(f"{v:.4f}" for v in tpose) + "\n")
        first = False
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
