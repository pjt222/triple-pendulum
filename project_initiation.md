# Triple Pendulum 3D Chaos Voxel Map

## Project Overview

Extend Drew's Campfire's 2D double pendulum chaos map to a **3D voxel visualization** of a triple pendulum system. In the original work, a 2D grid of initial conditions (θ₁, θ₂) is simulated and each pixel colored by chaos metric (e.g. time-to-first-flip). By moving to a triple pendulum we gain a third angle (θ₃), which naturally maps to a 3D volume — each **voxel** is an initial condition triplet (θ₁, θ₂, θ₃), colored by the same chaos metric.

**This appears to be a genuinely novel visualization.** No existing implementation of a 3D initial-condition-space chaos map for a chain pendulum was found in any public repo or publication.

## Motivation & Context

This project was inspired by:
- **Drew's Campfire** ([repo](https://github.com/drewscampfire/Drew-s-Campfire-Videos), [YouTube](https://www.youtube.com/@drewscampfire)) — 2D chaos maps of double pendulum using Manim + Blender, with GPU-batched simulation via `torchdiffeq` on CUDA
- A prior triple pendulum project by this repo's author extending Drew's 2D chaos maps to 3D voxel grids with an interactive Three.js renderer

## Physics

### Triple Pendulum Equations of Motion

For an n-link pendulum with equal point masses (m=1) and equal rod lengths (l=1):

- **Mass matrix:** `M_{ij} = a_{ij} * cos(θ_i - θ_j)` where `a_{ij} = n - max(i,j)`
- **Force vector:** `f_i = Σ_j a_{ij} * sin(θ_i - θ_j) * ω_j² + (n-i) * g * sin(θ_i)`
- **EOM:** `M * α = -f` (solve for angular accelerations α at each timestep)

For n=3, the coupling matrix is:
```
A = [[3, 2, 1],
     [2, 2, 1],
     [1, 1, 1]]
```

Gravity weights: `[3, 2, 1]`, g = 9.81.

The state vector is `[θ₁, θ₂, θ₃, ω₁, ω₂, ω₃]` (6D). All simulations start from rest (ω = 0), so the initial condition space is 3D: (θ₁, θ₂, θ₃).

### Alternative: Symbolic Derivation

For more complex configurations (unequal masses/lengths), use SymPy's Kane's method following Jake VanderPlas's approach (see references). The hardcoded equal-mass formulation above is sufficient for the visualization and much faster.

## Working Prototype

A working proof-of-concept exists with the following components:

### Simulation (`triple_pendulum_sim.py`)
- Vectorized numpy batch simulation of N triple pendulums
- RK4 integration with `np.linalg.solve` for the mass matrix at each step
- 40³ = 64,000 pendulums simulated in ~103 seconds on CPU
- Computes time-to-first-flip (angle wraps past ±180°) as the chaos metric
- Output: JSON with 3D grid of flip times
- **Result:** 83.2% of initial conditions flipped within 15s simulation time

### Visualization (`triple_pendulum_viz.html`)
- Three.js point cloud with additive blending
- Drag-to-orbit, scroll-to-zoom camera controls
- Per-axis slice controls (fix θ₁, θ₂, or θ₃ to see 2D cross-sections)
- Time range filter, point size/opacity controls
- Toggle stable vs. flipped voxels
- Magma-inspired colormap (dark purple → red → orange → yellow → white)

## Roadmap

### Phase 1: GPU-Accelerated Simulation
- Port simulation to **PyTorch + torchdiffeq** following Drew's `OptimizedDoublePendulumComputation` pattern
- Use CUDA batching with `odeint(method='dopri8')` for GPU-parallel integration
- Memory-mapped output with batch processing for large grids
- Target: **200³ = 8M voxels** (currently 40³ = 64K)
- Estimated GPU time: minutes on a modern NVIDIA GPU (vs. hours on CPU)

### Phase 2: High-Quality Volume Rendering
- Move from Three.js point cloud to proper **volumetric rendering**
- Options: Vispy (jonnyhyman/Chaos approach), Blender voxel rendering, or WebGPU ray marching
- Key insight from Softology: rendering all voxels gives a white blob — need to render **boundary/edge regions only** or use transparency/isosurfaces
- Implement isosurface extraction at specific flip-time thresholds
- Animated 2D slice sweeps (fix one angle, sweep through its range)

### Phase 3: Extended Analysis
- Multiple chaos metrics beyond flip-time: Lyapunov exponent, trajectory divergence rate, energy transfer patterns
- Fractal dimension estimation of the 3D boundary surfaces
- Comparison: how does the 3D structure relate to the known 2D double pendulum boundaries?
- Interactive real-time exploration with adaptive resolution (coarse far away, fine near boundaries)

### Phase 4: Publication / Media
- Blender-rendered flythrough of the 3D fractal structure
- Manim-animated explanation video
- Potential YouTube content or academic visualization paper

## Prior Art & References

### Direct Inspiration
| Project | Description | Relevance |
|---------|-------------|-----------|
| [drewscampfire/Drew-s-Campfire-Videos](https://github.com/drewscampfire/Drew-s-Campfire-Videos) | 2D double pendulum chaos maps, Manim + Blender, GPU-batched via torchdiffeq | Core approach we're extending to 3D |
| [jonnyhyman/Chaos](https://github.com/jonnyhyman/Chaos) (1.9k★) | 1000³ voxel volume rendering of Mandelbrot↔logistic map for Veritasium. Python + Vispy + Numba | Best open-source reference for high-res 3D chaos voxel rendering |
| [Softology / Visions of Chaos](https://softologyblog.wordpress.com/2019/02/09/mitchell-gravity-set-fractals/) | 3D magnetic pendulum basins + Mitchell Gravity Set at 500³. Volumetric rendering | Closest conceptual match (3D basins of attraction), but closed-source and different physical system |

### Pendulum Simulation References
| Project | Description |
|---------|-------------|
| [Jake VanderPlas triple pendulum blog](https://jakevdp.github.io/blog/2017/03/08/triple-pendulum-chaos/) | Triple pendulum via SymPy Kane's method, n-pendulum generalization |
| [PeterJochem/TriplePendulum](https://github.com/PeterJochem/TriplePendulum) | Lagrangian triple pendulum sim, trajectory divergence visualization |
| [kriskda/MagneticPendulumBasins](https://github.com/kriskda/MagneticPendulumBasins) | CUDA-accelerated 2D magnetic pendulum basins — good reference for GPU batching pattern |
| [vidanchev/Double_Chaos](https://github.com/vidanchev/Double_Chaos) | Double pendulum with C Dormand-Prince + Python viz (2D and 3D animations) |
| [beltoforion.de magnetic pendulum](https://beltoforion.de/en/magnetic_pendulum/) | TypeScript magnetic pendulum basins with excellent physics documentation |

### Visualization Frameworks
| Project | Description |
|---------|-------------|
| [gboeing/pynamical](https://github.com/gboeing/pynamical) | Python package for 3D phase diagrams of discrete dynamical systems |
| [gboeing/lorenz-system](https://github.com/gboeing/lorenz-system) | Lorenz attractor visualization, scipy + matplotlib |

### Key Visualization Challenges (from Softology)
- Naive rendering of all voxels produces an opaque blob — interior structure is hidden
- Effective approaches: slice removal (cut away octants), boundary-only rendering (show edges between basins), volumetric transparency with billboard quads, isosurface extraction
- For our case, the slice controls in the prototype already address this well for exploration; high-quality rendering needs one of the above approaches

## Technical Stack

### Current (prototype)
- Python 3 + NumPy (vectorized batch simulation)
- RK4 integration, `np.linalg.solve` for mass matrix
- Three.js r128 (WebGL point cloud visualization)
- JSON data interchange

### Target (production)
- **Simulation:** Python + PyTorch + torchdiffeq (CUDA GPU batching)
- **Data:** NumPy memmap for large grids (following Drew's pattern)
- **Visualization (interactive):** Vispy or Three.js/WebGPU
- **Visualization (rendered):** Blender volumetric or Python + matplotlib/Vispy for publication figures
- **Animation:** Manim (optional, for explanatory content)

## File Structure

```
triple-pendulum-chaos/
├── CLAUDE.md                  # This file
├── src/
│   ├── simulation/
│   │   ├── physics.py         # Triple pendulum EOM (numpy + torch versions)
│   │   ├── batch_sim.py       # Batch RK4/dopri8 simulation
│   │   └── metrics.py         # Chaos metrics (flip time, Lyapunov, etc.)
│   ├── visualization/
│   │   ├── viewer.html        # Interactive Three.js/WebGPU viewer
│   │   ├── volume_render.py   # Vispy volumetric rendering
│   │   └── colormap.py        # Colormaps and data→color transforms
│   └── utils/
│       ├── grid.py            # Initial condition grid construction
│       └── io.py              # Data I/O (JSON, memmap, HDF5)
├── data/                      # Generated simulation data (gitignored)
├── notebooks/                 # Exploration notebooks
└── renders/                   # Output images/videos
```

## Quick Start

```bash
# Run the prototype simulation (CPU, ~2 min for 40³)
python triple_pendulum_sim.py

# Open the visualization
# (serve triple_pendulum_viz.html or open directly in browser)
```

## Key Design Decisions

1. **Equal masses and lengths** — simplifies the coupling matrix to integer coefficients, making the physics clean and the visualization symmetric. Can be generalized later.
2. **Time-to-first-flip as primary metric** — same as Drew's approach, produces striking fractal boundaries between "easy to flip" and "hard to flip" regions.
3. **Released from rest** — all ω=0 initially, keeping the initial condition space to exactly 3 dimensions (θ₁, θ₂, θ₃).
4. **Angle range ±170°** — covers nearly all of configuration space while avoiding the singular point at exactly ±180°.
5. **Additive blending in viewer** — creates a plasma/nebula aesthetic where dense regions glow, naturally highlighting the fractal boundaries.

## License

CC BY-NC-SA 4.0 (matching Drew's Campfire repo license).