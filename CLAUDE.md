# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Triple Pendulum 3D Chaos Voxel Map — extends 2D double pendulum chaos maps (Drew's Campfire) to a 3D voxel visualization of a triple pendulum system. Each voxel represents an initial condition triplet (θ₁, θ₂, θ₃), colored by chaos metric (time-to-first-flip). All simulations start from rest (ω=0), so the 6D state space reduces to 3D initial conditions.

License: CC BY-NC-SA 4.0.

## Physics

Triple pendulum with equal point masses (m=1) and equal rod lengths (l=1):
- Mass matrix: `M_{ij} = a_{ij} * cos(θ_i - θ_j)` with coupling matrix `A = [[3,2,1],[2,2,1],[1,1,1]]`
- Force vector: `f_i = Σ_j a_{ij} * sin(θ_i - θ_j) * ω_j² + (n-i) * g * sin(θ_i)`, g=9.81
- Solve `M * α = -f` for angular accelerations at each timestep
- Angle range: ±170° (avoids singularity at ±180°)
- Flip detection: angle wraps past ±180°

## Architecture

```
src/
├── simulation/
│   ├── physics.py         # Triple pendulum EOM (numpy + torch + numba)
│   ├── cuda_sim.py        # CUDA C kernel via CuPy/PyCUDA
│   ├── batch_sim.py       # Batch RK4 simulation (CPU + GPU paths)
│   └── metrics.py         # Chaos metrics (flip time, Lyapunov)
├── visualization/
│   ├── viewer.html        # Interactive Three.js/WebGPU viewer
│   ├── volume_render.py   # Vispy volumetric rendering
│   └── colormap.py        # Colormaps and data→color transforms
└── utils/
    ├── grid.py            # Initial condition grid construction
    ├── io.py              # Data I/O (JSON, binary, memmap, HDF5)
    ├── convert.py         # Format conversion between storage backends
    └── registry.py        # Data registry generator (YAML inventory)
```

`data/` (gitignored) holds generated simulation output + `_registry.yml` inventory. `renders/` holds output images/videos.

## Tech Stack

- **Simulation (CPU):** Python 3 + NumPy + Numba JIT, RK4 integration, Cramer's rule
- **Simulation (GPU):** CuPy CUDA C kernel, ~170K pendulums/sec
- **Data formats:** JSON + binary (.bin + .meta.json) for zero-parse browser loading
- **Interactive viz:** Three.js (WebGL point cloud with additive blending)
- **Volume rendering:** Vispy or Blender volumetric
- **Animation:** Manim (optional)

## Realms

Two grid types (realms) for sampling initial conditions:

- **Cube** (default): Uniform Cartesian grid in (θ₁, θ₂, θ₃) space. N³ total points.
- **Sphere**: Concentric Fibonacci-spiral shells inscribed in the cube (r_max = 170°). ~(π/6)·N³ total points (~48% fewer). Enables radial chaos transition mapping and directional asymmetry analysis.

Sphere grid uses explicit `positions` array in JSON (avoids duplicating Fibonacci algorithm in JS). The simulation kernel is grid-agnostic — it receives (N, 3) initial angles regardless of realm.

## Resolution Grid

17 resolutions: `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 300, 400, 500]`

- Fine: 10-100 by 10
- Medium: 125, 150, 175, 200
- Coarse: 300, 400, 500

GPU (CuPy) runs all 17 resolutions for both realms (34 datasets).
CPU (Numba) runs 10-200 for both realms (28 datasets).
Total: 62 datasets tracked in `data/_registry.yml`.

## Commands

```bash
# GPU simulation (all resolutions, cube realm)
python3 run_simulations.py --backend gpu --realm cube

# GPU simulation (sphere realm)
python3 run_simulations.py --backend gpu --realm sphere

# CPU simulation (10-200 only, cube realm)
python3 run_simulations.py --backend cpu --realm cube

# Specific resolutions
python3 run_simulations.py --backend gpu --realm cube --resolutions 100 200

# Dry run (preview what would be simulated)
python3 run_simulations.py --dry-run

# Update data registry
python3 -m src.utils.registry

# Serve the interactive viewer locally
python3 serve.py
# then open http://localhost:8000/
```

## Key Design Constraints

- Naive rendering of all voxels produces an opaque blob. Use boundary-only rendering, slice removal, volumetric transparency, or isosurface extraction.
- The additive-blending point cloud in the viewer naturally highlights fractal boundaries.
- Slice controls (fix one θ axis) give 2D cross-sections for exploration.
- Max resolution: 500³ = 125M voxels (GPU). CPU capped at 200³ = 8M voxels.
- Sphere realm: `positions` array stores explicit point coordinates to avoid JS-side Fibonacci reimplementation; size overhead is offset by ~48% fewer points.
