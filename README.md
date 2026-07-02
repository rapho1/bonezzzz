# Bonezzzz — video → skeletal animation

A Blender add-on that turns a video of human motion into baked skeletal
animation, using MediaPipe or WHAM (SMPL) pose estimation. Everything happens
inside Blender — open the video, pick a backend, run, and the animated
armature lands directly in your scene.

> Previously this shipped as a separate Tauri desktop app with its own node
> editor. That's retired in favor of a proper Blender add-on — see
> [`app_legacy/`](app_legacy/) if you need to reference it. The engine that
> does the actual work is unchanged either way.

## Architecture

```
Bonezzzz/
├── pipeline/                # core: extract_pose, smooth, to_bvh, preview, verify_fk
├── engine/                  # graph compute engine + FastAPI server (JSON over HTTP)
│   ├── nodes.py              # node compute functions (wrap pipeline)
│   ├── graph.py               # DAG executor: per-node cache + heavy-node gate
│   ├── backends.py             # pluggable pose backends (MediaPipe, WHAM)
│   ├── exporters.py             # BVH/FBX export
│   └── server.py                # POST /run /preview /save, GET /health (port 8731)
└── blender_addon/bonezzzz/  # the add-on — this is the UI now
    ├── __init__.py            # bl_info, register()/unregister()
    ├── engine_process.py       # spawns/health-checks/kills the engine
    ├── engine_client.py         # HTTP calls to the engine
    ├── properties.py             # Scene-attached settings + graph builder
    ├── operators.py               # Open video / Run / Import / Export
    ├── panel.py                    # the sidebar UI
    └── bin/                          # bundled engine exe (built by build_sidecar.sh)
```

Execution is **hybrid**: cheap nodes (Smooth) recompute automatically; the heavy
Pose estimation node only runs on an explicit Run press, and its result is cached
to `cache/pose_<hash>.json` so re-runs are instant.

## Install & use

See **[blender_addon/INSTALL.md](blender_addon/INSTALL.md)** for the full
install/use guide. Short version: zip `blender_addon/bonezzzz/`, install it in
Blender via Preferences → Add-ons → Install from Disk, enable it, then use the
**Bonezzzz** tab in the 3D viewport sidebar (press `N` if hidden).

## Development (running from source)

```bash
# one-time: create the venv and install deps
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# point Blender at the add-on source instead of a zip:
#   Preferences > Add-ons > Install from Disk > blender_addon/bonezzzz/__init__.py
# (or symlink/copy blender_addon/bonezzzz/ into Blender's addons folder)
```

With no bundled exe present, `engine_process.py` falls back to launching
`.venv/Scripts/python.exe -m engine.server` directly — no need to rebuild the
sidecar for every code change while iterating.

## CLI (no Blender)

```bash
.venv/Scripts/python.exe pipeline/run.py test_videos/clip.mp4 output/clip.bvh
.venv/Scripts/python.exe pipeline/preview.py output/clip.bvh          # montage PNG
.venv/Scripts/python.exe pipeline/preview.py output/clip.bvh --gif    # animated GIF
```

## Status

- [x] Pose → smooth → BVH pipeline (FK-verified, < 0.05° error)
- [x] Graph engine with caching + heavy-node gating
- [x] Pluggable pose backends (`engine/backends.py`) — MediaPipe + WHAM both live
- [x] WHAM backend (SMPL) running in WSL2, GPU-accelerated
- [x] BVH / FBX export
- [x] **Full Blender add-on UI** (`blender_addon/bonezzzz/`) — replaces the old Tauri app
- [x] Engine bundled as a standalone exe the add-on auto-spawns (no venv needed at runtime)
- [ ] Blender Extensions platform packaging (`blender_manifest.toml`) — currently just a plain zip

## Packaging the engine

The add-on needs a standalone engine exe at
`blender_addon/bonezzzz/bin/bonezzzz-engine-x86_64-pc-windows-msvc.exe`.
Build/refresh it with:

```bash
./build_sidecar.sh
```

This PyInstalls `engine/server.py` (onefile, bundles MediaPipe/OpenCV/FastAPI)
and copies it into place. Re-run it whenever `engine/` or `pipeline/` changes,
then re-zip the add-on. WHAM stays external (WSL) regardless — it isn't part
of the bundled exe.

## WHAM backend (high-quality SMPL pose)

Optional, higher-quality pose backend — see **[WHAM_SETUP.md](WHAM_SETUP.md)**
for the full one-time setup (WSL2, GPU, license-gated SMPL models). Short
version: WHAM runs in a WSL2 Ubuntu conda env (`wham`, Python 3.9,
torch 1.11/cu113) because its `mmcv==1.3.9`/`chumpy` deps don't build on
native Windows. The engine's `wham` backend (`engine/backends.py`) shells into
WSL and runs `wham_to_bvh.py` (copied into the WSL WHAM checkout by
`setup_wham.sh`), which does detection → ViTPose → WHAM SMPL inference and
writes both a keypoints JSON (preview) and a BVH built directly from WHAM's
**exact SMPL joint angles** (export quality — this, not the geometric landmark
solver MediaPipe goes through, is what gives WHAM its real advantage). Peak
VRAM ~3 GB; WSL needs a bumped memory limit (see `setup_wslconfig.ps1`) or it
gets OOM-killed mid-run.

## Export formats

**BVH** is written directly by the engine. **FBX** export happens
**in-process inside the add-on's own Blender session** (`operators.py`'s
`BONEZZZZ_OT_export`: import the BVH into a scratch object, export just that
selection, clean up) — no external Blender process needed for the add-on's
export path. The engine's `/save?format=fbx` path (via a *separate, headless*
Blender subprocess in `engine/exporters.py`) still exists as a general engine
capability but the add-on doesn't call it.

## Pose backends

`engine/backends.py` is a registry of pose estimators returning the same
landmark format, so the rest of the graph is backend-agnostic. MediaPipe is
always available; WHAM needs the one-time setup in
[WHAM_SETUP.md](WHAM_SETUP.md).
