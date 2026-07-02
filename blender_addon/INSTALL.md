# Bonezzzz — Blender add-on

Video → skeletal animation, entirely inside Blender. No separate app.

## Install

1. Zip the `bonezzzz/` folder (the folder itself, so the zip contains
   `bonezzzz/__init__.py` etc. at its top level) — or use a pre-built
   `bonezzzz.zip` if you have one.
2. In Blender (4.2+): `Edit → Preferences → Add-ons`.
3. Top-right dropdown (▾) → `Install from Disk…` → pick the zip.
4. Tick the checkbox next to **Bonezzzz** to enable it.

That's it — no separate engine install. The add-on bundles a standalone
engine exe (`bonezzzz/bin/bonezzzz-engine-x86_64-pc-windows-msvc.exe`) and
starts/stops it automatically when the add-on is enabled/disabled.

## Use

Open the **3D viewport sidebar** (press `N` if it's hidden) → **Bonezzzz** tab:

1. **Open Video…** — pick a clip.
2. Choose a **backend** — MediaPipe works immediately; WHAM needs a one-time
   setup (see `WHAM_SETUP.md`) but gives noticeably better quality.
3. **Run Pose Estimation** — MediaPipe takes seconds, WHAM can take a couple
   of minutes on a new clip (cached afterward).
4. **Import Into Scene** to get an animated armature directly in your current
   scene, or pick a **format** (BVH/FBX) and **Export…** to a file instead.

## How it works

- `bonezzzz/engine_process.py` spawns the bundled engine exe on `register()`
  (or falls back to the project's `.venv` if you're running the add-on
  straight from source during development) and stops it on `unregister()`.
- The panel (`panel.py`) builds the same 4-node graph
  (`video_input → pose_estimation → smooth → output`) the engine's
  `/run`/`/save` endpoints expect, exactly like the project's earlier
  standalone-app frontend did — no engine changes were needed for this add-on.
- **Run** happens on a background thread with a modal timer polling for
  completion, so Blender's UI never freezes during a multi-minute WHAM run.
- **Import** and **FBX export** call `bpy.ops.import_anim.bvh` /
  `bpy.ops.export_scene.fbx` directly, in-process — no separate "Send to
  Blender" server or headless Blender subprocess needed, since the add-on
  already runs inside a live Blender session.

## Rebuilding the bundled engine

If you change `engine/` or `pipeline/`, rebuild and reinstall the bundled exe:

```bash
./build_sidecar.sh
```

This writes `bonezzzz/bin/bonezzzz-engine-x86_64-pc-windows-msvc.exe`. Re-zip
and reinstall the add-on (or just restart Blender if you're developing from
an unpacked source checkout — Blender picks up the new exe on the next
`register()`).

## Troubleshooting

- **"Engine: error: ..."** in the panel → the bundled exe failed to start, or
  no `.venv` was found for a source checkout. Check Blender's system console
  (`Window → Toggle System Console` on Windows) for the engine's own output.
- **WHAM backend errors** → see `WHAM_SETUP.md` — it needs a one-time WSL2 +
  conda + license-gated SMPL model setup.
- **FBX export fails** → this now happens entirely inside your own Blender
  session (no external Blender process needed), so a failure here is a real
  export error — check the error text in Blender's status bar / info log.
