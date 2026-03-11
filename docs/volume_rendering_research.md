# Volume Rendering Approach Research

Issue #21: Evaluate rendering approaches for the 3D chaos voxel map.

## Approaches Evaluated

### 1. Vispy (Python, Desktop)
- **Reference:** jonnyhyman/Chaos (1.9k stars) — rendered 1000^3 for Veritasium
- **Pros:** Python-native, proven at extreme scale, GPU-accelerated via OpenGL, supports Volume visual with custom colormaps and transfer functions
- **Cons:** Desktop-only (no browser), requires OpenGL context, limited interactivity compared to web
- **Performance:** Handles 200^3 at interactive framerates; 1000^3 with careful memory management
- **Status:** Implemented in `src/visualization/vispy_viewer.py`

### 2. Three.js Point Cloud (WebGL, Browser)
- **Reference:** Current prototype viewer
- **Pros:** Browser-based, no install needed, additive blending creates natural plasma aesthetic, wide compatibility
- **Cons:** Not true volumetric rendering, memory limits around 200^3 (16M points), no ray marching
- **Performance:** Smooth at 40^3 (64K points), acceptable at 100^3, struggles at 200^3
- **Status:** Implemented in `src/visualization/viewer.html`

### 3. Blender Volumetric (Offline Render)
- **Pros:** Publication-quality output, true volumetric with subsurface scattering, animation support, compositing
- **Cons:** Batch-only (no interactivity), slow render times, requires Blender expertise
- **Performance:** N/A (offline renderer, minutes per frame)
- **Status:** Planned for Phase 4

### 4. WebGPU Ray Marching (Browser)
- **Pros:** True volumetric in browser, modern API, compute shaders for direct volume rendering
- **Cons:** Limited browser support (Chrome 113+), newer ecosystem, more complex implementation
- **Performance:** Potentially handles 200^3 with ray marching, untested
- **Status:** Not implemented (future consideration)

## Recommendation

**Use a tiered approach:**

1. **Exploration:** Three.js point cloud (browser, instant, good for up to 100^3)
2. **Analysis:** Vispy volumetric (desktop, handles 200^3, boundary rendering, screenshots)
3. **Publication:** Blender volumetric (offline, highest quality, animation)

The key insight from Softology remains: naive full-volume rendering produces an opaque blob.
Effective techniques implemented:
- Boundary-only rendering (gradient threshold) — `volume_render.py`
- Isosurface extraction (marching cubes) — `volume_render.py`
- Octant slice removal — `viewer.html`
- Per-axis slice controls — `viewer.html`
- Additive blending point cloud — `viewer.html`

## Files

| Approach | File | Status |
|----------|------|--------|
| Three.js | `src/visualization/viewer.html` | Complete |
| Vispy | `src/visualization/vispy_viewer.py` | Complete |
| Boundary detection | `src/visualization/volume_render.py` | Complete |
| Slice animations | `src/visualization/slice_animation.py` | Complete |
| Adaptive resolution | `src/visualization/adaptive.py` | Complete |
| Blender | Phase 4 | Planned |
