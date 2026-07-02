bl_info = {
    "name": "Bonezzzz",
    "author": "Bonezzzz",
    "version": (0, 2, 4),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Bonezzzz",
    "description": "Video to skeletal animation (MediaPipe / WHAM) directly inside Blender.",
    "category": "Animation",
}

from . import engine_process
from . import properties
from . import operators
from . import panel

MODULES = (properties, operators, panel)


def register():
    for m in MODULES:
        m.register()
    engine_process.ensure_started()


def unregister():
    engine_process.stop()
    for m in reversed(MODULES):
        m.unregister()
