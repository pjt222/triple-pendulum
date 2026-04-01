#!/usr/bin/env python3
"""Blender script: render triple pendulum chaos isosurface as a turntable GIF.

Usage (called by render_hero.sh, not directly):
    blender --background --python scripts/render_hero.py

Reads the middle isosurface OBJ from renders/_meshes/ (produced by
extract_meshes.py), imports it into Blender with vertex colors derived
from local surface curvature, applies a cyberpunk-themed material, and
renders a turntable animation.
"""

import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MESH_DIR = PROJECT_ROOT / "renders" / "_meshes"
FRAME_DIR = PROJECT_ROOT / "renders" / "_frames"

RESOLUTION_X = 960
RESOLUTION_Y = 540
NUM_FRAMES = 36
SAMPLES = 48
CAMERA_DISTANCE = 480
CAMERA_ELEVATION = 22  # degrees above horizon

# Cyberpunk colormap stops: (normalized_position, R, G, B)
CYBERPUNK_STOPS = [
    (0.000, 0.040, 0.000, 0.080),
    (0.150, 0.000, 0.120, 0.300),
    (0.350, 0.000, 0.940, 1.000),
    (0.550, 1.000, 0.330, 0.870),
    (0.750, 1.000, 0.867, 0.000),
    (1.000, 1.000, 1.000, 1.000),
]


def lerp_colormap(t: float) -> tuple[float, float, float]:
    """Interpolate the cyberpunk colormap at position t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    for i in range(len(CYBERPUNK_STOPS) - 1):
        pos_a, ra, ga, ba = CYBERPUNK_STOPS[i]
        pos_b, rb, gb, bb = CYBERPUNK_STOPS[i + 1]
        if t <= pos_b:
            frac = (t - pos_a) / (pos_b - pos_a) if pos_b > pos_a else 0
            return (
                ra + frac * (rb - ra),
                ga + frac * (gb - ga),
                ba + frac * (bb - ba),
            )
    return CYBERPUNK_STOPS[-1][1:]


def clear_scene():
    """Remove all default objects."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def apply_vertex_colors(obj):
    """Color vertices by normalized position (height) using the cyberpunk colormap."""
    mesh = obj.data

    # Get bounding box for normalization
    verts = [v.co for v in mesh.vertices]
    min_vals = Vector((min(v.x for v in verts), min(v.y for v in verts), min(v.z for v in verts)))
    max_vals = Vector((max(v.x for v in verts), max(v.y for v in verts), max(v.z for v in verts)))
    extent = max_vals - min_vals
    extent = Vector((max(e, 0.001) for e in extent))

    # Create vertex color layer
    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name="ChaosColor")
    color_layer = mesh.vertex_colors["ChaosColor"]

    # Color each loop vertex based on its normalized position
    # Use a combination of position axes to get varied coloring
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            co = mesh.vertices[vert_idx].co
            # Map 3D position to colormap: use radial distance + height mix
            nx = (co.x - min_vals.x) / extent.x
            ny = (co.y - min_vals.y) / extent.y
            nz = (co.z - min_vals.z) / extent.z
            # Blend of axes for interesting color variation
            t = 0.3 * nx + 0.3 * ny + 0.4 * nz
            r, g, b = lerp_colormap(t)
            color_layer.data[loop_idx].color = (r, g, b, 1.0)

    return color_layer


def create_vertex_color_material(name: str):
    """Create a Principled BSDF material that reads vertex colors."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Vertex color node
    vcol = nodes.new("ShaderNodeVertexColor")
    vcol.layer_name = "ChaosColor"

    # Principled BSDF
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = 0.3
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["IOR"].default_value = 1.45
    # Slight emission for glow
    bsdf.inputs["Emission Strength"].default_value = 0.15

    # Output
    output = nodes.new("ShaderNodeOutputMaterial")

    # Connect vertex color to base color and emission
    links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(vcol.outputs["Color"], bsdf.inputs["Emission Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat


def import_main_isosurface(level_info: list[dict]):
    """Import the middle isosurface (most fractal detail) and apply vertex colors."""
    # Pick the middle surface (index 2 of 5)
    mid_idx = len(level_info) // 2
    info = level_info[mid_idx]

    obj_path = MESH_DIR / info["obj_file"]
    if not obj_path.exists():
        print(f"ERROR: {obj_path} not found")
        sys.exit(1)

    print(f"  Importing {obj_path.name} ({info['num_vertices']:,} verts)...")
    bpy.ops.wm.obj_import(filepath=str(obj_path))
    obj = bpy.context.selected_objects[0]
    obj.name = "ChaosSurface"

    # Smooth shading
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()

    # Apply vertex colors
    print("  Applying vertex colors...")
    apply_vertex_colors(obj)

    # Apply material
    mat = create_vertex_color_material("ChaosMaterial")
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    return obj


def setup_camera_turntable():
    """Create camera on a turntable rig for 360-degree rotation."""
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

    # Animate 360-degree rotation
    pivot.rotation_euler = (0, 0, 0)
    pivot.keyframe_insert(data_path="rotation_euler", frame=1)
    pivot.rotation_euler = (0, 0, math.radians(360))
    pivot.keyframe_insert(data_path="rotation_euler", frame=NUM_FRAMES + 1)

    if pivot.animation_data and pivot.animation_data.action:
        for fcurve in pivot.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = "LINEAR"

    return camera, pivot


def setup_lighting():
    """Three-point cyberpunk lighting."""
    # Key light: cyan-tinted from upper left
    bpy.ops.object.light_add(type="AREA", location=(-300, -400, 350))
    key = bpy.context.object
    key.name = "KeyLight"
    key.data.energy = 80000
    key.data.color = (0.6, 0.9, 1.0)
    key.data.size = 150
    key_constraint = key.constraints.new("TRACK_TO")
    key_constraint.target = bpy.data.objects.get("CameraPivot")
    key_constraint.track_axis = "TRACK_NEGATIVE_Z"
    key_constraint.up_axis = "UP_Y"

    # Fill light: magenta from right
    bpy.ops.object.light_add(type="AREA", location=(350, 200, 100))
    fill = bpy.context.object
    fill.name = "FillLight"
    fill.data.energy = 30000
    fill.data.color = (1.0, 0.3, 0.8)
    fill.data.size = 200
    fill_constraint = fill.constraints.new("TRACK_TO")
    fill_constraint.target = bpy.data.objects.get("CameraPivot")
    fill_constraint.track_axis = "TRACK_NEGATIVE_Z"
    fill_constraint.up_axis = "UP_Y"

    # Rim light: warm yellow from behind
    bpy.ops.object.light_add(type="AREA", location=(0, 400, 200))
    rim = bpy.context.object
    rim.name = "RimLight"
    rim.data.energy = 50000
    rim.data.color = (1.0, 0.85, 0.4)
    rim.data.size = 120
    rim_constraint = rim.constraints.new("TRACK_TO")
    rim_constraint.target = bpy.data.objects.get("CameraPivot")
    rim_constraint.track_axis = "TRACK_NEGATIVE_Z"
    rim_constraint.up_axis = "UP_Y"


def setup_render():
    """Configure Cycles render settings."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False

    # Try GPU
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

    # Dark world background
    world = bpy.data.worlds.new("ChaosWorld")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Color"].default_value = (0.005, 0.0, 0.015, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0
    scene.render.film_transparent = False


def main():
    levels_path = MESH_DIR / "levels.json"
    if not levels_path.exists():
        print(f"ERROR: {levels_path} not found. Run extract_meshes.py first.")
        sys.exit(1)

    with open(levels_path) as f:
        metadata = json.load(f)

    print(f"Rendering hero GIF from {metadata['resolution']}^3 data")

    print("Setting up scene...")
    clear_scene()

    print("Importing isosurface...")
    obj = import_main_isosurface(metadata["levels"])

    print("Setting up camera...")
    setup_camera_turntable()

    print("Setting up lighting...")
    setup_lighting()

    print("Configuring render...")
    setup_render()

    print(f"Rendering {NUM_FRAMES} frames at {RESOLUTION_X}x{RESOLUTION_Y}...")
    bpy.ops.render.render(animation=True)

    print(f"\nFrames saved to {FRAME_DIR}/")


if __name__ == "__main__":
    main()
