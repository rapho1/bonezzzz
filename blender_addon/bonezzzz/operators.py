"""
Operators: pick a video (native Blender file browser), run pose estimation
(non-blocking modal + background thread), import the result into the current
scene, and export BVH/FBX.

Non-blocking pattern for Run: the HTTP call happens on a plain threading.Thread
so Blender's UI never freezes (WHAM can take minutes); the thread only ever
touches plain Python objects (self._result/self._error), never bpy data —
bpy property writes happen exclusively on the main thread in _finish(), which
runs from the modal's TIMER handling.
"""
import os
import threading

import bpy
from bpy_extras.io_utils import ImportHelper

from . import engine_client
from . import engine_process
from .properties import graph_from_props


class BONEZZZZ_OT_pick_video(bpy.types.Operator, ImportHelper):
    bl_idname = "bonezzzz.pick_video"
    bl_label = "Open Video"

    filter_glob: bpy.props.StringProperty(
        default="*.mp4;*.mov;*.avi;*.mkv;*.webm", options={'HIDDEN'})

    def execute(self, context):
        context.scene.bonezzzz.video_path = self.filepath
        return {'FINISHED'}


class BONEZZZZ_OT_run_pose(bpy.types.Operator):
    bl_idname = "bonezzzz.run_pose"
    bl_label = "Run Pose Estimation"

    _timer = None
    _thread = None
    _result = None
    _error = None

    def invoke(self, context, event):
        props = context.scene.bonezzzz
        if props.busy:
            self.report({'WARNING'}, "A run is already in progress.")
            return {'CANCELLED'}
        if not props.video_path:
            self.report({'ERROR'}, "Pick a video first.")
            return {'CANCELLED'}

        # The startup poll timer can give up (e.g. slow first launch of a
        # freshly-installed exe) even though the engine comes up moments
        # later. ensure_started() is a cheap no-op if it's already healthy,
        # so re-check here rather than leaving a stale "error" status forever.
        if not engine_process.STATE["status"] == "ready":
            engine_process.ensure_started()
            if engine_process.STATE["status"] != "ready":
                self.report({'ERROR'}, f"Engine: {engine_process.STATE['status']}")
                return {'CANCELLED'}

        props.busy = True
        props.has_result = False
        props.status_text = (
            "Running WHAM - can take minutes on a new clip..."
            if props.backend == 'wham' else "Running...")

        self._result = None
        self._error = None
        graph = graph_from_props(props)

        def worker():
            try:
                self._result = engine_client.run(graph, "o", allow_heavy=True)
            except Exception as e:  # noqa: BLE001 - surface any failure to the panel
                self._error = str(e)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.3, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER' and not self._thread.is_alive():
            self._finish(context)
            return {'FINISHED'}
        return {'PASS_THROUGH'}

    def _finish(self, context):
        props = context.scene.bonezzzz
        context.window_manager.event_timer_remove(self._timer)
        props.busy = False
        if self._error:
            props.status_text = f"Error: {self._error}"
            self.report({'ERROR'}, self._error)
        else:
            statuses = (self._result or {}).get("statuses", {})
            info = statuses.get("o", {}).get("info", "done")
            props.status_text = f"Done - {info}"
            props.has_result = True
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class BONEZZZZ_OT_clear_cache(bpy.types.Operator):
    bl_idname = "bonezzzz.clear_cache"
    bl_label = "Clear Cache"
    bl_description = (
        "Drop cached pose-estimation results (in-memory and on disk under "
        "cache/). Use this if Run keeps producing an old animation after "
        "changing the video or add-on settings")

    def execute(self, context):
        props = context.scene.bonezzzz
        try:
            res = engine_client.clear_cache()
        except Exception as e:  # noqa: BLE001
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        props.has_result = False
        props.status_text = f"Cache cleared ({res.get('disk_files_removed', 0)} file(s))."
        return {'FINISHED'}


class BONEZZZZ_OT_import_scene(bpy.types.Operator):
    bl_idname = "bonezzzz.import_scene"
    bl_label = "Import Into Scene"

    def execute(self, context):
        props = context.scene.bonezzzz
        graph = graph_from_props(props)
        tmp_path = os.path.join(bpy.app.tempdir, "bonezzzz_import.bvh")
        try:
            engine_client.save(graph, "o", tmp_path, "bvh")
        except Exception as e:  # noqa: BLE001
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        try:
            bpy.ops.import_anim.bvh(
                filepath=tmp_path, global_scale=0.01,
                use_fps_scale=True, update_scene_fps=False)
        except Exception as e:  # noqa: BLE001
            self.report({'ERROR'}, f"Blender import failed: {e}")
            return {'CANCELLED'}
        props.status_text = "Imported into scene."
        return {'FINISHED'}


class BONEZZZZ_OT_export(bpy.types.Operator):
    bl_idname = "bonezzzz.export"
    bl_label = "Export Animation"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    format: bpy.props.EnumProperty(
        items=[('bvh', "BVH", ""), ('fbx', "FBX", "")], default='bvh')

    def invoke(self, context, event):
        props = context.scene.bonezzzz
        self.format = props.export_format
        base = "animation"
        if props.video_path:
            base = os.path.splitext(os.path.basename(props.video_path))[0]
        self.filepath = base + "." + self.format
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        props = context.scene.bonezzzz
        graph = graph_from_props(props)

        if self.format == 'bvh':
            try:
                engine_client.save(graph, "o", self.filepath, "bvh")
            except Exception as e:  # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            props.status_text = f"Saved -> {self.filepath}"
            return {'FINISHED'}

        # FBX: fetch BVH to a temp file, import as a scratch object, export
        # only that selection (sidesteps Blender's FBX-exporter light-object
        # bug and leaves the user's existing scene untouched), then clean up.
        tmp_bvh = os.path.join(bpy.app.tempdir, "bonezzzz_export_tmp.bvh")
        try:
            engine_client.save(graph, "o", tmp_bvh, "bvh")
        except Exception as e:  # noqa: BLE001
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        before = set(context.scene.objects)
        try:
            bpy.ops.import_anim.bvh(
                filepath=tmp_bvh, global_scale=0.01,
                use_fps_scale=True, update_scene_fps=False)
        except Exception as e:  # noqa: BLE001
            self.report({'ERROR'}, f"Import failed: {e}")
            return {'CANCELLED'}
        new_objs = [o for o in context.scene.objects if o not in before]

        bpy.ops.object.select_all(action='DESELECT')
        for o in new_objs:
            o.select_set(True)
        try:
            bpy.ops.export_scene.fbx(
                filepath=self.filepath, use_selection=True,
                add_leaf_bones=False, bake_anim=True,
                bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
                bake_anim_use_all_actions=False,
            )
            props.status_text = f"Saved -> {self.filepath}"
        except Exception as e:  # noqa: BLE001
            self.report({'ERROR'}, f"FBX export failed: {e}")
            return {'CANCELLED'}
        finally:
            for o in new_objs:
                bpy.data.objects.remove(o, do_unlink=True)

        return {'FINISHED'}


CLASSES = (
    BONEZZZZ_OT_pick_video,
    BONEZZZZ_OT_run_pose,
    BONEZZZZ_OT_clear_cache,
    BONEZZZZ_OT_import_scene,
    BONEZZZZ_OT_export,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
