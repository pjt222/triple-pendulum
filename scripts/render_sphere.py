#!/usr/bin/env python3
"""Blender script: render sphere realm chaos isosurface as a turntable GIF.

Usage:
    python3 scripts/prep_sphere.py       # resample + extract isosurface
    blender --background --python scripts/render_sphere.py

Identical approach to render_hero.py but for the sphere realm isosurface.
"""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MESH_DIR = PROJECT_ROOT / "renders" / "_meshes"
FRAME_DIR = PROJECT_ROOT / "renders" / "_sphere_frames"

RESOLUTION_X = 960
RESOLUTION_Y = 540
NUM_FRAMES = 72
SAMPLES = 48
CAMERA_DISTANCE = 1100
CAMERA_ELEVATION = 25

CYBERPUNK_STOPS = [
    (0.000, 0.040, 0.000, 0.080),
    (0.150, 0.000, 0.120, 0.300),
    (0.350, 0.000, 0.940, 1.000),
    (0.550, 1.000, 0.330, 0.870),
    (0.750, 1.000, 0.867, 0.000),
    (1.000, 1.000, 1.000, 1.000),
]


def lerp_colormap(t):
    t = max(0.0, min(1.0, t))
    for i in range(len(CYBERPUNK_STOPS) - 1):
        p0, r0, g0, b0 = CYBERPUNK_STOPS[i]
        p1, r1, g1, b1 = CYBERPUNK_STOPS[i + 1]
        if t <= p1:
            f = (t - p0) / (p1 - p0) if p1 > p0 else 0
            return (r0 + f * (r1 - r0), g0 + f * (g1 - g0), b0 + f * (b1 - b0))
    return CYBERPUNK_STOPS[-1][1:]


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def apply_vertex_colors(obj):
    """Color by angular position for varied coloring on a spherical surface.

    Uses a blend of elevation (theta3/z) and azimuth to create a smooth
    gradient that wraps around the sphere.
    """
    mesh = obj.data

    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name="ChaosColor")
    color_layer = mesh.vertex_colors["ChaosColor"]

    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            co = mesh.vertices[vert_idx].co
            # Normalize each axis to [0, 1] from [-170, 170]
            nx = (co.x + 170) / 340
            ny = (co.y + 170) / 340
            nz = (co.z + 170) / 340
            # Blend axes for varied angular coloring
            t = 0.35 * nx + 0.30 * ny + 0.35 * nz
            r, g, b = lerp_colormap(t)
            color_layer.data[loop_idx].color = (r, g, b, 1.0)

    return color_layer


def create_vertex_color_material(name):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    vcol = nodes.new("ShaderNodeVertexColor")
    vcol.layer_name = "ChaosColor"

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = 0.3
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["IOR"].default_value = 1.45
    bsdf.inputs["Emission Strength"].default_value = 0.4

    output = nodes.new("ShaderNodeOutputMaterial")

    links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(vcol.outputs["Color"], bsdf.inputs["Emission Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat


def setup_camera_turntable():
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    pivot = bpy.context.object
    pivot.name = "CameraPivot"

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "HeroCamera"

    elevation_rad = math.radians(CAMERA_ELEVATION)
    camera.location = (
        CAMERA_DISTANCE * math.cos(elevation_rad),
        0,
        CAMERA_DISTANCE * math.sin(elevation_rad),
    )
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = pivot
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    camera.parent = pivot
    bpy.context.scene.camera = camera

    pivot.rotation_euler = (0, 0, 0)
    pivot.keyframe_insert(data_path="rotation_euler", frame=1)
    pivot.rotation_euler = (0, 0, math.radians(360))
    pivot.keyframe_insert(data_path="rotation_euler", frame=NUM_FRAMES + 1)

    if pivot.animation_data and pivot.animation_data.action:
        for fcurve in pivot.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = "LINEAR"


def setup_lighting():
    bpy.ops.object.light_add(type="AREA", location=(-300, -400, 350))
    key = bpy.context.object
    key.name = "KeyLight"
    key.data.energy = 80000
    key.data.color = (0.6, 0.9, 1.0)
    key.data.size = 150
    c = key.constraints.new("TRACK_TO")
    c.target = bpy.data.objects.get("CameraPivot")
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"

    bpy.ops.object.light_add(type="AREA", location=(350, 200, 100))
    fill = bpy.context.object
    fill.name = "FillLight"
    fill.data.energy = 30000
    fill.data.color = (1.0, 0.3, 0.8)
    fill.data.size = 200
    c = fill.constraints.new("TRACK_TO")
    c.target = bpy.data.objects.get("CameraPivot")
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"

    bpy.ops.object.light_add(type="AREA", location=(0, 400, 200))
    rim = bpy.context.object
    rim.name = "RimLight"
    rim.data.energy = 50000
    rim.data.color = (1.0, 0.85, 0.4)
    rim.data.size = 120
    c = rim.constraints.new("TRACK_TO")
    c.target = bpy.data.objects.get("CameraPivot")
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False

    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs:
        try:
            prefs.preferences.compute_device_type = "CUDA"
            prefs.preferences.get_devices()
            for device in prefs.preferences.devices:
                device.use = True
            scene.cycles.device = "GPU"
            print("  Using CUDA GPU rendering")
        except Exception:
            print("  Falling back to CPU rendering")

    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.resolution_percentage = 100

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(FRAME_DIR / "frame_")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    scene.frame_start = 1
    scene.frame_end = NUM_FRAMES

    world = bpy.data.worlds.new("ChaosWorld")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Color"].default_value = (0.005, 0.0, 0.015, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0
    scene.render.film_transparent = False


def main():
    obj_path = MESH_DIR / "sphere_iso.obj"
    meta_path = MESH_DIR / "sphere_iso.json"

    if not obj_path.exists():
        print(f"ERROR: {obj_path} not found. Run scripts/prep_sphere.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Rendering sphere isosurface: {meta['num_vertices']:,} vertices")

    clear_scene()

    print("Importing mesh...")
    bpy.ops.wm.obj_import(filepath=str(obj_path))
    obj = bpy.context.selected_objects[0]
    obj.name = "SphereChaos"

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()

    print("Applying vertex colors...")
    apply_vertex_colors(obj)

    mat = create_vertex_color_material("SphereChaosMat")
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    setup_camera_turntable()
    setup_lighting()
    setup_render()

    print(f"Rendering {NUM_FRAMES} frames at {RESOLUTION_X}x{RESOLUTION_Y}...")
    bpy.ops.render.render(animation=True)

    print(f"\nFrames saved to {FRAME_DIR}/")


if __name__ == "__main__":
    main()
