"""
FastAPI server exposing the graph engine to the Bonezzzz Blender add-on.

Endpoints:
  POST /run          {graph, target, allow_heavy}   -> statuses + summary
  POST /preview      {graph, target}                -> skeleton preview payload
  POST /save         {graph, target, path, format}  -> writes .bvh/.fbx to disk
  POST /cache/clear  {}                              -> drops all cached results
  GET  /health
"""
import glob
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.graph import Engine
from engine.nodes import CACHE_DIR
from engine.preview_payload import build_preview
from engine import exporters

app = FastAPI(title="Bonezzzz engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Engine()


class RunReq(BaseModel):
    graph: dict
    target: str
    allow_heavy: bool = False


class PreviewReq(BaseModel):
    graph: dict
    target: str


class SaveReq(BaseModel):
    graph: dict
    target: str
    path: str
    format: str = "bvh"


# Bumped on every engine change the add-on must not silently miss.
# blender_addon/bonezzzz/engine_process.py keeps a matching constant and
# replaces any running engine that reports a different (or no) version -
# otherwise "Install from Disk" updates leave a stale engine process serving
# old math, and new options are silently ignored.
ENGINE_VERSION = "0.2.4"


@app.get("/health")
def health():
    return {"ok": True, "version": ENGINE_VERSION}


@app.post("/run")
def run(req: RunReq):
    try:
        return engine.execute(req.graph, req.target, req.allow_heavy)
    except Exception as e:  # noqa: BLE001 - surface as clean API error
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/preview")
def preview(req: PreviewReq):
    res = engine.get_result(req.graph, req.target)
    if res is None or "frames" not in res:
        return {"fps": None, "names": [], "bones": [], "frames": []}
    return build_preview(res["frames"], res.get("fps", 30.0))


@app.post("/cache/clear")
def clear_cache():
    """Drop both cache layers: the in-memory per-node cache (stale results
    otherwise survive as long as this engine process keeps running - e.g.
    across Blender restarts if a leftover process is still up) and the
    on-disk pose cache under CACHE_DIR (stale results otherwise survive even
    a fresh engine process, since heavy nodes skip recompute when a matching
    pose_<key>.json already exists on disk)."""
    engine.clear_cache()
    removed = 0
    for path in glob.glob(os.path.join(CACHE_DIR, "pose_*.json")):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return {"ok": True, "disk_files_removed": removed}


@app.post("/save")
def save(req: SaveReq):
    res = engine.get_result(req.graph, req.target)
    if res is None or res.get("kind") != "bvh":
        raise HTTPException(status_code=400,
                            detail="Output node has no baked result. Run it first.")
    try:
        path = exporters.export(res["bvh"], req.path, req.format)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "path": path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8731)
