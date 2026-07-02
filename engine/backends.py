"""
Pluggable pose-estimation backends. Each backend takes a video path + params
and returns (frames, fps), where frames is a list of per-frame landmark lists
(or None) in MediaPipe-world layout (33 landmarks, x/y/z/visibility).

Adding a backend = register a callable here. The rest of the graph (smooth,
to_bvh, export) is backend-agnostic because everything downstream consumes the
same landmark format.
"""
import json
import os
import subprocess
import threading

from pipeline.extract_pose import extract_pose

WSL_DISTRO = "Ubuntu"
WHAM_DIR = "/root/WHAM"
WHAM_TIMEOUT_S = 1200  # 20 min hard cap for a WHAM run
_wham_lock = threading.Lock()  # serialize WHAM: one GPU run at a time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cwd-independent


def _mediapipe(video_path: str, params: dict):
    frames, fps = extract_pose(
        video_path,
        model_complexity=int(params.get("complexity", 2)),
        min_detection_confidence=float(params.get("min_detection", 0.5)),
        min_tracking_confidence=float(params.get("min_tracking", 0.5)),
    )
    return frames, fps, None  # (frames, fps, angle_bvh)


def _win_to_wsl(path: str) -> str:
    """C:\\Users\\x\\v.mp4 -> /mnt/c/Users/x/v.mp4"""
    ap = os.path.abspath(path)
    drive = ap[0].lower()
    return "/mnt/" + drive + ap[2:].replace("\\", "/")


def _wham(video_path: str, params: dict):
    """
    Runs WHAM inside WSL (conda env `wham`) via wham_to_bvh.py, which does
    detection -> ViTPose -> WHAM SMPL inference, then produces BOTH:
      - an angle-based BVH straight from the SMPL joint rotations (real WHAM
        quality; used for export), and
      - a keypoints JSON (SMPL joints -> our landmarks; used for preview/graph).
    All torch/GPU work stays in WSL; this process orchestrates and reads files.
    Returns (frames, fps, angle_bvh_text).
    """
    wsl_video = _win_to_wsl(os.path.join(REPO_ROOT, video_path)
                            if not os.path.isabs(video_path) else video_path)
    # Use a fixed Windows-side cache dir for round-tripping results; this is
    # independent of where this engine process itself is running from (so it
    # works the same whether launched via `python -m engine.server` or as the
    # packaged/frozen sidecar exe).
    cache_root = os.environ.get("BONEZZZZ_DATA_DIR", REPO_ROOT)
    cache = os.path.join(cache_root, "output", "wham_cache")
    os.makedirs(cache, exist_ok=True)
    out_json_win = os.path.join(cache, "keypoints.json")
    out_bvh_win = os.path.join(cache, "wham.bvh")
    wsl_json, wsl_bvh = map(_win_to_wsl, (out_json_win, out_bvh_win))

    # wham_to_bvh.py is expected to live INSIDE the WHAM checkout in WSL
    # (copied there once during setup — see WHAM_SETUP.md) rather than being
    # resolved from this process's own location, which would break once this
    # engine is packaged into a standalone/frozen exe.
    inner = (
        f"source /root/miniconda3/etc/profile.d/conda.sh && conda activate wham && "
        f"cd {WHAM_DIR} && PYTHONPATH={WHAM_DIR} python '{WHAM_DIR}/wham_to_bvh.py' "
        f"--video '{wsl_video}' --out_bvh '{wsl_bvh}' --out_json '{wsl_json}' "
        f"--out_dir output/wham_cache"
    )
    # Serialize so repeated requests can't stack multiple GPU runs (VRAM OOM).
    if not _wham_lock.acquire(blocking=False):
        raise RuntimeError("A WHAM run is already in progress. Wait for it to finish.")
    try:
        for stale in (out_json_win, out_bvh_win):
            if os.path.exists(stale):
                os.remove(stale)
        try:
            proc = subprocess.run(
                ["wsl", "-d", WSL_DISTRO, "bash", "-lc", inner],
                capture_output=True, text=True, timeout=WHAM_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"WHAM timed out after {WHAM_TIMEOUT_S // 60} min. The clip may be "
                "too long/large for this GPU, or a previous run left it busy.")
        if not os.path.exists(out_json_win) or not os.path.exists(out_bvh_win):
            tail = (proc.stdout + proc.stderr)[-800:]
            raise RuntimeError(
                "WHAM (WSL) did not produce output. Is WSL/Ubuntu running and the "
                f"`wham` env set up?\n{tail}")
        with open(out_json_win) as f:
            d = json.load(f)
        with open(out_bvh_win) as f:
            bvh = f.read()
        return d["frames"], float(d["fps"]), bvh
    finally:
        _wham_lock.release()


BACKENDS = {
    "mediapipe": _mediapipe,
    "wham": _wham,
}


def run_backend(name: str, video_path: str, params: dict):
    fn = BACKENDS.get(name)
    if fn is None:
        raise ValueError(f"unknown pose backend: {name!r}")
    return fn(video_path, params)
