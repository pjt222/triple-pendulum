# Triple Pendulum 3D Chaos Voxel Map

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
**[Live Interactive Viewer](https://huggingface.co/spaces/pjt222/triple-pendulum)** on Hugging Face Spaces

![Fractal cross-sections of the triple pendulum chaos map at six θ₃ slices](renders/slice_gallery_theta3.png)

Map the chaos of a triple pendulum by sweeping all possible starting angles (θ₁, θ₂, θ₃) and measuring how long each configuration takes to flip. All three pendulums start from rest, so the full 6D phase space collapses to a 3D grid of initial conditions -- each voxel colored by its time-to-first-flip. The result is a fractal structure that extends Drew's Campfire's [2D double pendulum chaos maps](https://github.com/drewscampfire/Drew-s-Campfire-Videos) into the third dimension.

![Animated sweep through θ₃ slices](renders/slice_sweep_theta3.gif)

## Quick Start

```bash
pip install -e .            # CPU only (NumPy + Numba)
pip install -e ".[gpu]"     # GPU support (CuPy CUDA)

python3 run_simulations.py --backend gpu --realm cube   # GPU simulation
python3 run_simulations.py --backend cpu --realm cube   # CPU simulation (10-200)
python3 serve.py                                        # viewer at http://localhost:8000
```

## Simulation

The simulator auto-selects the fastest available backend:

| Backend | 40³ (64K) | 200³ (8M) | 600³ (216M) |
|---------|-----------|-----------|-------------|
| CUDA C (CuPy) | 0.5s | 46s | ~25 min |
| Numba JIT | 41s | ~6 min | -- |
| NumPy | 120s | ~20 min | -- |

Backend priority: **CuPy > PyCUDA > PyTorch GPU > Numba > NumPy** (auto-detected at runtime).

Resolution grid: `[10, 20, 30, ..., 100, 125, 150, 175, 200, 300, 400, 500, 600]` (18 steps). GPU runs all 18; CPU capped at 200. Two grid geometries (realms):

```bash
python3 run_simulations.py --backend gpu --realm cube     # uniform Cartesian (default)
python3 run_simulations.py --backend gpu --realm sphere   # Fibonacci-spiral shells (~48% fewer points)
python3 run_simulations.py --dry-run                      # preview what would run
```

## Viewer

The [interactive viewer](https://huggingface.co/spaces/pjt222/triple-pendulum) is a Three.js point cloud with additive blending that naturally highlights fractal boundaries.

- **Slice controls** -- fix any θ axis to explore 2D cross-sections
- **6 colormaps** -- cyberpunk (default), magma, viridis, inferno, plasma, cividis
- **Adaptive time filter** -- auto-narrows to keep ~2M visible points at high resolutions
- **Auto-rotate** with adjustable speed, axes toggle, and colormap invert

## Project Structure

```
src/
  simulation/       Physics engine, batch RK4, CUDA C kernel
  visualization/    Colormaps and rendering tools
  utils/            Grid construction, data I/O, binary format, registry
docs/               Interactive viewer (index.html + simulation data)
data/               Simulation output + _registry.yml inventory (gitignored)
renders/            Images and animations
```

## Prior Art

- [Drew's Campfire](https://github.com/drewscampfire/Drew-s-Campfire-Videos) -- 2D double pendulum chaos maps
- [jonnyhyman/Chaos](https://github.com/jonnyhyman/Chaos) -- 1000³ voxel rendering for Veritasium
- [Jake VanderPlas](https://jakevdp.github.io/blog/2017/03/08/triple-pendulum-chaos/) -- Triple pendulum via SymPy

## License

CC BY-NC-SA 4.0
