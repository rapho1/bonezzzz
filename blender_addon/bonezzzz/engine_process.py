"""
Finds/spawns/health-checks/kills the Bonezzzz Python engine.

Mirrors what app_legacy/src-tauri/src/lib.rs did in Rust: prefer a bundled
standalone exe (bin/bonezzzz-engine-*.exe, built by build_sidecar.sh), fall
back to the project's .venv for local development. No bpy.types classes here,
so this module isn't registered like the others — __init__.py calls
ensure_started()/stop() directly.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import bpy

HOST = "127.0.0.1"
PORT = 8731
HEALTH_URL = f"http://{HOST}:{PORT}/health"
# Must match ENGINE_VERSION in engine/server.py (baked into the bundled exe).
# A reachable engine reporting a different version is a leftover from an
# older add-on install and gets replaced - otherwise updating the add-on
# keeps serving old math and silently ignores new options.
ENGINE_VERSION = "0.2.4"
POLL_INTERVAL = 1.0
# A freshly-installed/extracted exe can take a while to pass antivirus
# scanning on its first launch from a new path - seen in practice taking
# longer than a plain 30s window right after "Install from Disk".
POLL_TIMEOUT = 90.0

# idle | starting | ready | error: <message>
STATE = {"status": "idle"}

_proc = None
_poll_started_at = None


def health_ok(timeout=1.0) -> bool:
    return engine_version(timeout) is not None


def engine_version(timeout=1.0) -> str | None:
    """Version string of a reachable engine; "0" for pre-versioning engines;
    None when nothing is listening."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as r:
            if r.status != 200:
                return None
            data = json.loads(r.read().decode("utf-8"))
            return str(data.get("version", "0"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _kill_port_owner():
    """Kill whatever process is listening on our port (Windows only) - used
    to replace a stale engine left over from a previous add-on version."""
    if sys.platform != "win32":
        return
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW).stdout
        pids = set()
        for line in out.splitlines():
            parts = line.split()
            if (len(parts) >= 5 and parts[0] == "TCP"
                    and parts[1].endswith(f":{PORT}")
                    and parts[3] == "LISTENING"):
                pids.add(parts[4])
        for pid in pids:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", pid],
                creationflags=subprocess.CREATE_NO_WINDOW,
                capture_output=True)
    except OSError:
        pass


def _bundled_exe_path():
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    exe = os.path.join(addon_dir, "bin", "bonezzzz-engine-x86_64-pc-windows-msvc.exe")
    return exe if os.path.exists(exe) else None


def _dev_venv_python():
    """blender_addon/bonezzzz/engine_process.py -> ../../.. = repo root."""
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(addon_dir))
    python = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
    return (python, repo_root) if os.path.exists(python) else (None, None)


def _redraw_view3d():
    try:
        for wm in bpy.data.window_managers:
            for win in wm.windows:
                for area in win.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
    except Exception:  # noqa: BLE001 - never let a redraw hiccup break polling
        pass


def _poll():
    if health_ok():
        STATE["status"] = "ready"
        _redraw_view3d()
        return None  # stop the timer
    if time.monotonic() - _poll_started_at[0] > POLL_TIMEOUT:
        STATE["status"] = "error: engine did not respond in time"
        _redraw_view3d()
        return None
    return POLL_INTERVAL


def ensure_started():
    """Idempotent — reuses a reachable engine of the RIGHT version; replaces
    a reachable engine of the wrong version (stale exe from an older add-on
    install that would silently serve outdated math)."""
    global _proc, _poll_started_at

    ver = engine_version()
    if ver == ENGINE_VERSION:
        STATE["status"] = "ready"
        return
    if ver is not None:
        _kill_port_owner()
        _proc = None

    STATE["status"] = "starting"
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    exe = _bundled_exe_path()
    try:
        if exe:
            _proc = subprocess.Popen([exe], creationflags=creationflags)
        else:
            python, repo_root = _dev_venv_python()
            if not python:
                STATE["status"] = "error: no bundled engine exe and no .venv found"
                return
            _proc = subprocess.Popen(
                [python, "-m", "engine.server"], cwd=repo_root,
                creationflags=creationflags)
    except OSError as e:
        STATE["status"] = f"error: failed to launch engine ({e})"
        return

    _poll_started_at = [time.monotonic()]
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll, first_interval=POLL_INTERVAL)


def stop():
    global _proc
    if _proc is not None:
        try:
            if sys.platform == "win32":
                # The bundled exe is a PyInstaller onefile bootloader: it
                # extracts itself and runs the real server as a CHILD process,
                # then waits on it. Popen.terminate() only signals the
                # bootloader, leaving the actual server running — kill the
                # whole tree instead.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(_proc.pid)],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    capture_output=True,
                )
            else:
                _proc.terminate()
        except OSError:
            pass
        _proc = None
    STATE["status"] = "idle"
