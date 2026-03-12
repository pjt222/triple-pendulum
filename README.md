# Triple Pendulum 3D Chaos Voxel Map

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**[Live Interactive Viewer](https://huggingface.co/spaces/pjt222/triple-pendulum)** (hosted on Hugging Face Spaces)

3D voxel visualization of triple pendulum chaos. Each voxel represents an initial condition triplet (theta_1, theta_2, theta_3), colored by time-to-first-flip. Extends the 2D double pendulum chaos maps from [Drew's Campfire](https://github.com/drewscampfire/Drew-s-Campfire-Videos) into the third dimension.

## Quick Start

```bash
# Install
pip install -e .

# Run CPU simulation (40^3 grid, ~2 min)
python -m src.simulation.batch_sim

# Serve interactive viewer
python -m http.server 8000
# Open http://localhost:8000/src/visualization/viewer.html
```

## Physics

Triple pendulum with equal point masses (m=1) and equal rod lengths (l=1):

- **Mass matrix:** M_ij = a_ij * cos(theta_i - theta_j), coupling A = [[3,2,1],[2,2,1],[1,1,1]]
- **Force vector:** f_i = sum_j a_ij * sin(theta_i - theta_j) * omega_j^2 + (n-i) * g * sin(theta_i)
- **EOM:** M * alpha = -f, solved for angular accelerations at each timestep
- All simulations start from rest (omega=0), reducing the 6D state space to 3D initial conditions

## Project Structure

```
src/
  simulation/     Physics engine and batch simulation
    physics.py    Triple pendulum equations of motion
    batch_sim.py  Batch RK4/dopri8 integration
    metrics.py    Chaos metrics (flip time, Lyapunov)
  visualization/  Rendering and interactive viewers
    viewer.html   Three.js point cloud viewer
    colormap.py   Colormaps and data-to-color transforms
  utils/          Grid construction and data I/O
    grid.py       Initial condition grid builder
    io.py         JSON, memmap, HDF5 I/O
data/             Simulation output (gitignored)
notebooks/        Exploration notebooks
renders/          Output images and videos
```

## Roadmap

- **Phase 0:** Project setup and CPU prototype
- **Phase 1:** GPU-accelerated simulation (PyTorch + torchdiffeq, target 200^3 = 8M voxels)
- **Phase 2:** High-quality volume rendering (boundary-only, isosurfaces, slice animations)
- **Phase 3:** Extended analysis (Lyapunov exponents, fractal dimension, energy transfer)
- **Phase 4:** Publication and media (Blender renders, Manim animations)

## Prior Art

- [Drew's Campfire](https://github.com/drewscampfire/Drew-s-Campfire-Videos) -- 2D double pendulum chaos maps
- [jonnyhyman/Chaos](https://github.com/jonnyhyman/Chaos) -- 1000^3 voxel rendering for Veritasium
- [Jake VanderPlas](https://jakevdp.github.io/blog/2017/03/08/triple-pendulum-chaos/) -- Triple pendulum via SymPy

## License

CC BY-NC-SA 4.0
