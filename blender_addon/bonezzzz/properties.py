"""
Scene-attached settings for the Bonezzzz panel, and the function that turns
them into the JSON graph the engine's /run and /save endpoints expect.

This mirrors the fixed 4-node graph the old Tauri frontend built in
app_legacy/src/store.ts's toApiGraph() — same node ids/types/param keys, so
engine/nodes.py and engine/backends.py needed zero changes for this add-on.
"""
import bpy


def graph_from_props(props):
    return {
        "nodes": [
            {"id": "v", "type": "video_input", "params": {
                "source_type": props.source_type,
                "path": props.video_path,
            }},
            {"id": "p", "type": "pose_estimation", "params": {
                "backend": props.backend,
                "complexity": int(props.complexity),
                "min_detection": props.min_detection,
                "min_tracking": props.min_tracking,
            }},
            {"id": "s", "type": "smooth", "params": {
                "cutoff": props.cutoff,
                "trim_edges": props.trim_edges,
            }},
            {"id": "o", "type": "output", "params": {
                "tpose_start": props.tpose_start,
            }},
        ],
        "edges": [
            {"source": "v", "target": "p"},
            {"source": "p", "target": "s"},
            {"source": "s", "target": "o"},
        ],
    }


class BonezzzzProperties(bpy.types.PropertyGroup):
    video_path: bpy.props.StringProperty(
        name="Video", description="Source video file",
        default="", subtype='FILE_PATH')
    source_type: bpy.props.EnumProperty(
        name="Source", items=[
            ('video', "Video file", ""),
            ('keypoints', "Keypoints JSON", ""),
        ], default='video')
    backend: bpy.props.EnumProperty(
        name="Backend", items=[
            ('mediapipe', "MediaPipe", "Fast, always available"),
            ('wham', "WHAM (SMPL, via WSL)", "Higher quality, needs one-time setup"),
        ], default='mediapipe')
    complexity: bpy.props.EnumProperty(
        name="Model Complexity", items=[
            ('0', "Lite (fast)", ""),
            ('1', "Full", ""),
            ('2', "Heavy (best)", ""),
        ], default='2')
    min_detection: bpy.props.FloatProperty(
        name="Min Detection", default=0.5, min=0.1, max=0.9)
    min_tracking: bpy.props.FloatProperty(
        name="Min Tracking", default=0.5, min=0.1, max=0.9)
    cutoff: bpy.props.FloatProperty(
        name="Cutoff Frequency (Hz)", default=6.0, min=1.0, max=12.0)
    trim_edges: bpy.props.BoolProperty(
        name="Trim Undetected Edges", default=True)
    tpose_start: bpy.props.BoolProperty(
        name="T-Pose Start Frame",
        description=(
            "Prepend one frame with the skeleton in its rest T-pose "
            "(zero rotations) - handy for binding/rigging a character "
            "before the animation starts. MediaPipe backend only"),
        default=False)
    export_format: bpy.props.EnumProperty(
        name="Format", items=[
            ('bvh', "BVH (.bvh)", ""),
            ('fbx', "FBX (.fbx)", ""),
        ], default='bvh')
    status_text: bpy.props.StringProperty(name="Status", default="Ready.")
    busy: bpy.props.BoolProperty(name="Busy", default=False)
    has_result: bpy.props.BoolProperty(name="Has Result", default=False)


CLASSES = (BonezzzzProperties,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.bonezzzz = bpy.props.PointerProperty(type=BonezzzzProperties)


def unregister():
    del bpy.types.Scene.bonezzzz
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
