"""VIEW_3D > Sidebar > Bonezzzz tab."""
import bpy

from . import engine_process


class VIEW3D_PT_bonezzzz(bpy.types.Panel):
    bl_label = "Bonezzzz"
    bl_idname = "VIEW3D_PT_bonezzzz"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Bonezzzz"

    def draw(self, context):
        layout = self.layout
        props = context.scene.bonezzzz
        status = engine_process.STATE["status"]

        icon = 'CHECKMARK' if status == 'ready' else (
            'ERROR' if status.startswith('error') else 'INFO')
        layout.label(text=f"Engine: {status}", icon=icon)

        col = layout.column(align=True)
        col.prop(props, "video_path", text="")
        col.operator("bonezzzz.pick_video", text="Open Video...")

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "backend")
        if props.backend == 'mediapipe':
            col.prop(props, "complexity")
            col.prop(props, "min_detection")
        else:
            box = layout.box()
            box.label(text="Needs one-time WHAM setup.", icon='ERROR')
            box.label(text="See WHAM_SETUP.md.")

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "cutoff")
        col.prop(props, "trim_edges")
        if props.backend == 'mediapipe':
            col.prop(props, "tpose_start")

        layout.separator()
        row = layout.row(align=True)
        row.enabled = not props.busy
        row.operator("bonezzzz.run_pose", text="Run Pose Estimation", icon='PLAY')
        sub = row.row(align=True)
        sub.enabled = not props.busy
        sub.operator("bonezzzz.clear_cache", text="", icon='TRASH')
        layout.label(text=props.status_text)

        layout.separator()
        col = layout.column(align=True)
        col.enabled = props.has_result and not props.busy
        col.operator("bonezzzz.import_scene", text="Import Into Scene")
        row = col.row(align=True)
        row.prop(props, "export_format", text="")
        row.operator("bonezzzz.export", text="Export...")


CLASSES = (VIEW3D_PT_bonezzzz,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
