"""
Thin, blocking HTTP client for the Bonezzzz engine. No bpy usage here on
purpose — callers (operators.py) are responsible for running these off
Blender's main thread when they might take a while (WHAM runs).
"""
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8731"


def _post(path, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            raise RuntimeError(json.loads(raw).get("detail", raw)) from None
        except (ValueError, KeyError):
            raise RuntimeError(raw or str(e)) from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Engine not reachable: {e.reason}") from None


def run(graph: dict, target: str, allow_heavy: bool) -> dict:
    # WHAM can take minutes; keep this above the engine's own 20-min WHAM
    # timeout so a slow-but-real run isn't killed client-side first.
    return _post("/run", {"graph": graph, "target": target, "allow_heavy": allow_heavy},
                timeout=1260)


def save(graph: dict, target: str, path: str, fmt: str) -> dict:
    return _post("/save", {"graph": graph, "target": target, "path": path, "format": fmt},
                timeout=180)
