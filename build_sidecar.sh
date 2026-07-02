#!/bin/bash
# Rebuilds the MediaPipe engine as a standalone exe and installs it into the
# Blender add-on package (blender_addon/bonezzzz/bin/). Run this whenever
# engine/ or pipeline/ code changes and you want the add-on's bundled engine
# to pick it up. The add-on auto-spawns this exe on register() — see
# blender_addon/bonezzzz/engine_process.py.
set -e
cd "$(dirname "$0")"

rm -rf build dist bonezzzz-engine.spec

.venv/Scripts/python.exe -m PyInstaller --noconfirm --onefile --console \
  --name bonezzzz-engine \
  --collect-all mediapipe \
  --collect-all cv2 \
  --hidden-import uvicorn.logging \
  --hidden-import uvicorn.loops \
  --hidden-import uvicorn.loops.auto \
  --hidden-import uvicorn.protocols \
  --hidden-import uvicorn.protocols.http \
  --hidden-import uvicorn.protocols.http.auto \
  --hidden-import uvicorn.protocols.websockets \
  --hidden-import uvicorn.protocols.websockets.auto \
  --hidden-import uvicorn.lifespan \
  --hidden-import uvicorn.lifespan.on \
  engine/server.py

mkdir -p blender_addon/bonezzzz/bin
cp dist/bonezzzz-engine.exe blender_addon/bonezzzz/bin/bonezzzz-engine-x86_64-pc-windows-msvc.exe

echo "Sidecar installed -> blender_addon/bonezzzz/bin/bonezzzz-engine-x86_64-pc-windows-msvc.exe"
