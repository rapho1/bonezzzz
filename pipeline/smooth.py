"""
Temporal smoothing of per-frame landmarks.
Fills missing frames via interpolation, then applies a Butterworth low-pass filter.
"""
import numpy as np
from scipy.signal import butter, filtfilt


NUM_LANDMARKS = 33
COORDS = ["x", "y", "z"]


def _butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    normal_cutoff = min(normal_cutoff, 0.99)  # must be < 1
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    # filtfilt needs at least padlen+1 samples; padlen default = 3*(max(len(a),len(b))-1)
    min_len = 3 * (max(len(a), len(b)) - 1) + 1
    if len(data) < min_len:
        return data
    return filtfilt(b, a, data)


def trim_undetected_edges(
    frames: list[list[dict] | None],
) -> tuple[list[list[dict] | None], int, int]:
    """Drop leading/trailing frames where no pose was detected.
    Returns (trimmed_frames, n_dropped_start, n_dropped_end)."""
    start = 0
    while start < len(frames) and frames[start] is None:
        start += 1
    end = len(frames)
    while end > start and frames[end - 1] is None:
        end -= 1
    return frames[start:end], start, len(frames) - end


def smooth_landmarks(
    frames: list[list[dict] | None],
    fps: float,
    cutoff_hz: float = 6.0,
    trim_edges: bool = False,
) -> list[list[dict]]:
    """
    frames: raw output from extract_pose (None where pose not detected)
    trim_edges: if True, drop leading/trailing undetected frames first.
    Returns: list of (possibly trimmed) frames, all filled, smoothed.
    """
    if trim_edges:
        frames, _, _ = trim_undetected_edges(frames)

    n_frames = len(frames)
    if n_frames == 0:
        return []

    # Build (n_frames, 33, 3) array; NaN where missing
    raw = np.full((n_frames, NUM_LANDMARKS, len(COORDS)), np.nan)
    vis = np.full((n_frames, NUM_LANDMARKS), 0.0)

    for i, frame in enumerate(frames):
        if frame is None:
            continue
        for j, lm in enumerate(frame):
            raw[i, j, 0] = lm["x"]
            raw[i, j, 1] = lm["y"]
            raw[i, j, 2] = lm["z"]
            vis[i, j] = lm["visibility"]

    t = np.arange(n_frames)

    # Interpolate NaN gaps per landmark per coord
    for j in range(NUM_LANDMARKS):
        for c in range(len(COORDS)):
            col = raw[:, j, c]
            mask = ~np.isnan(col)
            if mask.sum() < 2:
                # Not enough data — fill with zeros
                raw[:, j, c] = 0.0
            elif mask.sum() < n_frames:
                raw[:, j, c] = np.interp(t, t[mask], col[mask])

    # Low-pass filter per landmark per coord
    for j in range(NUM_LANDMARKS):
        for c in range(len(COORDS)):
            raw[:, j, c] = _butter_lowpass(raw[:, j, c], cutoff_hz, fps)

    # Also smooth visibility
    for j in range(NUM_LANDMARKS):
        vis[:, j] = _butter_lowpass(vis[:, j], cutoff_hz, fps)

    # Reconstruct frames list
    smoothed: list[list[dict]] = []
    for i in range(n_frames):
        frame_lms = []
        for j in range(NUM_LANDMARKS):
            frame_lms.append({
                "x": float(raw[i, j, 0]),
                "y": float(raw[i, j, 1]),
                "z": float(raw[i, j, 2]),
                "visibility": float(np.clip(vis[i, j], 0.0, 1.0)),
            })
        smoothed.append(frame_lms)

    return smoothed
