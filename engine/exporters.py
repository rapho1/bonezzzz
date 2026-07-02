"""
Export baked BVH to other formats. BVH is written directly; FBX is
produced by driving a headless Blender (its exporter is the industry standard).
"""
import glob
import os
import subprocess
import tempfile

# Maps the UI format to (extension, blender fmt token). bvh handled separately.
FORMATS = {
    "bvh": (".bvh", None),
    "fbx": (".fbx", "fbx"),
}

_CONVERT_SCRIPT = os.path.join(os.path.dirname(__file__), "blender_convert.py")


def find_blender() -> str | None:
    env = os.environ.get("BLENDER_PATH")
    if env and os.path.exists(env):
        return env
    patterns = [
        r"C:\Program Files\Blender Foundation\*\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\*\blender.exe",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
    ]
    hits = []
    for p in patterns:
        hits.extend(glob.glob(p))
    return sorted(hits)[-1] if hits else None


def export(bvh_text: str, out_path: str, fmt: str) -> str:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format: {fmt}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if fmt == "bvh":
        with open(out_path, "w") as f:
            f.write(bvh_text)
        return os.path.abspath(out_path)

    blender = find_blender()
    if not blender:
        raise RuntimeError(
            "Blender not found. Install Blender (blender.org) or set the "
            "BLENDER_PATH environment variable to export FBX. "
            "BVH export always works without Blender.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bvh", mode="w") as tmp:
        tmp.write(bvh_text)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [blender, "-b", "--factory-startup", "--python", _CONVERT_SCRIPT,
             "--", tmp_path, os.path.abspath(out_path), FORMATS[fmt][1]],
            capture_output=True, text=True, timeout=180,
        )
        if "CONVERT_OK" not in result.stdout or not os.path.exists(out_path):
            tail = (result.stdout + result.stderr)[-500:]
            raise RuntimeError(f"Blender export failed:\n{tail}")
        return os.path.abspath(out_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
