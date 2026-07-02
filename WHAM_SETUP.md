# Setting up the WHAM backend

Bonezzzz works out of the box with the **MediaPipe** pose backend — no setup
needed. **WHAM** is an optional, higher-quality backend (real SMPL joint
angles, no foot skating) that needs a one-time setup because of two hard
constraints that can't be automated away:

1. WHAM's dependencies (`mmcv==1.3.9`, `chumpy`) don't have Windows wheels and
   won't build natively on Windows — it has to run inside **WSL2 Ubuntu**.
2. The SMPL body models it needs are **license-gated**. Their license forbids
   redistribution, so every user must register and download them individually
   — the add-on cannot ship them for you.

If either of those is a dealbreaker, just use MediaPipe — it's fully
supported and needs nothing extra.

## Requirements

- Windows 10 (2004+) or Windows 11
- An NVIDIA GPU with **8 GB+ VRAM** and a recent driver (WHAM peaks around 3
  GB VRAM in practice, but ViTPose-Huge's checkpoint alone is 2.5 GB)
- **16 GB+ system RAM** recommended (WHAM's own process can use 7+ GB)
- ~15 GB free disk space (checkpoints + conda env + WHAM repo)
- ~30–60 minutes, most of it unattended downloading/compiling

## Step 1 — Enable WSL2

Skip this if `wsl -l -v` already shows an Ubuntu distro on version 2.

Open **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu
```

Reboot when it asks. On first launch, Ubuntu will ask you to create a
username/password — any values are fine, they're local to WSL only.

Verify GPU access from inside WSL before continuing:

```bash
nvidia-smi
```

If that fails, update your Windows NVIDIA driver (CUDA-on-WSL support is
driver-side, no separate Linux driver needed).

## Step 2 — Give WSL enough memory

From a normal (non-admin) **Windows PowerShell**, in this repo folder:

```powershell
powershell -ExecutionPolicy Bypass -File setup_wslconfig.ps1
```

This writes `%USERPROFILE%\.wslconfig` with a generous memory/swap allocation
and restarts WSL. Without this, WHAM gets OOM-killed partway through a run on
machines with 16 GB RAM or less.

## Step 3 — Run the automated setup

From **inside WSL** (open the "Ubuntu" app, or `wsl -d Ubuntu` from
PowerShell), navigate to this repo (it's visible under `/mnt/c/...`) and run:

```bash
cd /mnt/c/Users/<you>/Bonezzzz   # adjust to wherever you cloned/copied this repo
bash setup_wham.sh
```

This installs, in order: apt build tools, Miniconda, a `wham` conda env
(Python 3.9), PyTorch 1.11+cu113, the WHAM repo and its Python dependencies
(including compiling `mmcv==1.3.9` and `chumpy` from source — this is the
step native Windows can't do), ViTPose, and all the *public* model
checkpoints (WHAM's own weights, YOLOv8, ViTPose-Huge — none of these need a
login). It's idempotent, so if it fails partway through (flaky download,
etc.), just re-run it.

## Step 4 — Get the SMPL models (the one manual step)

Register at both of these (same account usually works for both — accept the
license on each site):

- https://smpl.is.tue.mpg.de
- https://smplify.is.tue.mpg.de

Then, still inside WSL:

```bash
cd /root/WHAM
bash fetch_smpl.sh
```

It'll ask for your email/password for the site above and place three `.pkl`
files under `dataset/body_models/smpl/`. If it fails with a "download looks
wrong" error, the script tells you the likely cause (bad credentials, or a
license not yet accepted on one of the two sites).

## Step 5 — Verify

In Blender's **Bonezzzz** sidebar tab (View3D > Sidebar, press `N` if hidden):
pick a short test clip, select **WHAM** as the backend, and press
**Run Pose Estimation**. First run on a new clip takes a couple of minutes
(detection + SMPL inference on the GPU); repeated runs on the same clip are
instant (cached). If it errors, re-check `nvidia-smi` inside WSL and that
`conda env list` shows `wham`.

## How it fits together

- `engine/backends.py`'s `wham` backend shells a command into WSL
  (`wsl -d Ubuntu bash -lc "..."`) that activates the `wham` conda env and
  runs `wham_to_bvh.py` (copied into `/root/WHAM` by `setup_wham.sh`).
- That script does detection (YOLO) → 2D pose (ViTPose-Huge) → SMPL inference
  (WHAM network) → writes both a keypoints JSON (used internally by the
  engine's graph) and a BVH built directly from WHAM's SMPL joint *angles*
  (what the Blender add-on actually imports/exports — this is what gives WHAM
  its real quality advantage over the geometric solver MediaPipe goes through).
- Only one WHAM run executes at a time (a lock in `backends.py`) — running it
  twice concurrently on an 8 GB GPU causes VRAM exhaustion and hangs.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `nvidia-smi` fails inside WSL | Update Windows NVIDIA driver; confirm `wsl -l -v` shows version 2 |
| WHAM run hangs / whole PC lags | Check `nvidia-smi` inside WSL for stuck processes (`kill -9` them); confirm `.wslconfig` was applied (`wsl --shutdown` then retry) |
| `mmcv`/`chumpy` build errors in `setup_wham.sh` | Re-run the script — these need `numpy==1.22.3`/`setuptools==59.5.0` installed first, which the script does, but a partial previous attempt can leave a bad state; `conda env remove -n wham` and re-run to start clean |
| SMPL download "looks wrong" / 500 error | Double check you accepted the license on **both** smpl.is.tue.mpg.de and smplify.is.tue.mpg.de |
| Panel says "Engine: error: ..." | The bundled engine exe failed to start (or no `.venv` was found for a source checkout) — this is unrelated to WHAM itself; see `blender_addon/INSTALL.md`'s troubleshooting section |
