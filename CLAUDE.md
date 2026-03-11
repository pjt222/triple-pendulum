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
│   ├── physics.py         # Triple pendulum EOM (numpy + torch)
│   ├── batch_sim.py       # Batch RK4/dopri8 simulation
│   └── metrics.py         # Chaos metrics (flip time, Lyapunov)
├── visualization/
│   ├── viewer.html        # Interactive Three.js/WebGPU viewer
│   ├── volume_render.py   # Vispy volumetric rendering
│   └── colormap.py        # Colormaps and data→color transforms
└── utils/
    ├── grid.py            # Initial condition grid construction
    └── io.py              # Data I/O (JSON, memmap, HDF5)
```

`data/` (gitignored) holds generated simulation output. `renders/` holds output images/videos.

## Tech Stack

- **Simulation (CPU):** Python 3 + NumPy, RK4 integration, `np.linalg.solve`
- **Simulation (GPU):** PyTorch + torchdiffeq, CUDA batching, `odeint(method='dopri8')`
- **Large data:** NumPy memmap for grids beyond memory
- **Interactive viz:** Three.js (WebGL point cloud with additive blending)
- **Volume rendering:** Vispy or Blender volumetric
- **Animation:** Manim (optional)

## Commands

```bash
# CPU simulation (~2 min for 40³ grid)
python triple_pendulum_sim.py

# Serve the interactive viewer
python -m http.server 8000
# then open http://localhost:8000/triple_pendulum_viz.html
```

## Key Design Constraints

- Naive rendering of all voxels produces an opaque blob. Use boundary-only rendering, slice removal, volumetric transparency, or isosurface extraction.
- The additive-blending point cloud in the viewer naturally highlights fractal boundaries.
- Slice controls (fix one θ axis) give 2D cross-sections for exploration.
- Target resolution: 200³ = 8M voxels (GPU), up from 40³ = 64K (CPU prototype).
