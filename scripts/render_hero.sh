#!/usr/bin/env bash
# Render a hero GIF of the triple pendulum chaos structure.
# Usage: bash scripts/render_hero.sh [--resolution 150] [--levels 5]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRAME_DIR="$PROJECT_ROOT/renders/_frames"
OUTPUT="$PROJECT_ROOT/renders/hero.gif"

echo "=== Step 1: Extract isosurface meshes ==="
python3 "$SCRIPT_DIR/extract_meshes.py" "$@"

echo ""
echo "=== Step 2: Render in Blender ==="
blender --background --python "$SCRIPT_DIR/render_hero.py" 2>&1 | grep -v "^$"

echo ""
echo "=== Step 3: Assemble GIF ==="
# Two-pass palette-optimized GIF via ffmpeg
ffmpeg -y -framerate 15 -i "$FRAME_DIR/frame_%04d.png" \
    -vf "palettegen=max_colors=128:stats_mode=diff" \
    "$FRAME_DIR/palette.png" 2>/dev/null

ffmpeg -y -framerate 15 -i "$FRAME_DIR/frame_%04d.png" \
    -i "$FRAME_DIR/palette.png" \
    -lavfi "paletteuse=dither=bayer:bayer_scale=3" \
    "$OUTPUT" 2>/dev/null

echo ""
echo "=== Done ==="
echo "Hero GIF: $OUTPUT"
echo "Size: $(du -h "$OUTPUT" | cut -f1)"
