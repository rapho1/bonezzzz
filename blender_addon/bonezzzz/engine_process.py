"""
Finds/spawns/health-checks/kills the Bonezzzz Python engine.

Mirrors what app_legacy/src-tauri/src/lib.rs did in Rust: prefer a bundled
standalone exe (bin/bonezzzz-engine-*.exe, built by build_sidecar.sh), fall
back to the project's .venv for local development. No bpy.types classes here,
so this module isn't registered like the others — __init__.py calls
ensure_started()/stop() directly.
"""
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
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


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
    """Idempotent — does nothing if an engine is already reachable, ours or not."""
    global _proc, _poll_started_at

    if health_ok():
        STATE["status"] = "ready"
        return

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
