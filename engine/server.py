"""
FastAPI server exposing the graph engine to the Bonezzzz Blender add-on.

Endpoints:
  POST /run      {graph, target, allow_heavy}       -> statuses + summary
  POST /preview  {graph, target}                    -> skeleton preview payload
  POST /save     {graph, target, path, format}       -> writes .bvh/.fbx to disk
  GET  /health
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.graph import Engine
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
ENGINE_VERSION = "0.2.3"


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
