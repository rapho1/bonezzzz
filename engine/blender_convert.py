"""
Runs INSIDE Blender (headless) to convert a BVH into FBX.

Invoked as:
    blender -b --factory-startup --python engine/blender_convert.py -- <in.bvh> <out> fbx
"""
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
bvh_in, out_path, fmt = argv[0], argv[1], argv[2]

bpy.ops.import_anim.bvh(
    filepath=bvh_in,
    global_scale=0.01,        # BVH is in centimetres
    use_fps_scale=True,
    update_scene_fps=False,
)

# Keep only the imported armature — drop the default camera/light/cube so the
# export is just the rig + animation (also avoids a Blender 4.3 FBX light bug).
for obj in list(bpy.data.objects):
    if obj.type != "ARMATURE":
        bpy.data.objects.remove(obj, do_unlink=True)

if fmt == "fbx":
    bpy.ops.export_scene.fbx(
        filepath=out_path,
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
    )
else:
    print(f"CONVERT_ERROR unknown format {fmt}")
    sys.exit(1)

print("CONVERT_OK", out_path)
