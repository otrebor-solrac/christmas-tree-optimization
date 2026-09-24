# 🎄 Christmas Tree Packing Optimization - Project Summary

## 📋 Overview
This project implements a comprehensive optimization pipeline for the **Christmas Tree Packing Problem**, where the goal is to pack N Christmas trees into the smallest possible square area while avoiding overlaps.

The solution combines multiple optimization algorithms implemented in **Rust** for performance, with **Python** scripts for batch processing and visualization.

---

## 🏆 Key Algorithms Used

### 1. **Simulated Annealing (SA)** ⭐ Most Used
The primary workhorse of the project. Simulated Annealing explores the solution space by:
- Making random perturbations (translations and rotations)
- Accepting improvements unconditionally
- Accepting worse solutions with probability based on temperature
- Gradually cooling down to converge to a local optimum

**Variants:**
- **SA Hot**: High temperature (0.001 → 0.00001) for broad exploration
- **SA Medium**: Medium temperature (0.0001 → 0.000001) for refinement
- **SA Cold**: Low temperature (0.00001 → 0.000000001) for fine-tuning

**Implementation:** `rust_optimizer/src/algorithms/annealing2.rs`

### 2. **Deca Strategy (Base Construction)**
Creates initial solutions by tiling optimized 10-tree blocks (`T10.csv`) in a grid pattern. This provides a strong starting point for further optimization.

**Implementation:** `rust_optimizer/src/algorithms/deca.rs`

### 3. **Genetic Algorithm (GA)**
Population-based evolutionary algorithm that:
- Maintains a population of candidate solutions
- Uses crossover and mutation operators
- Selects the fittest individuals for reproduction
- Supports `--pure-random` mode for exploration from scratch

**Implementation:** `rust_optimizer/src/algorithms/genetic.rs`

### 4. **Fine-Tuning**
Makes small, greedy adjustments to tree positions and rotations:
- Tries small perturbations in all directions
- Only accepts improvements
- Very effective for polishing near-optimal solutions

**Implementation:** `rust_optimizer/src/algorithms/fine_tuning.rs`

### 5. **Gravity Compression**
Pushes all trees toward the center (0,0) while maintaining validity:
- Iteratively moves trees inward
- Stops when collisions would occur
- Reduces the bounding box size

**Implementation:** `rust_optimizer/src/algorithms/gravity.rs`

### 6. **Rim Pressure**
Applies inward pressure from the boundary:
- Identifies trees on the perimeter
- Pushes them toward the center
- Helps compact the overall arrangement

**Implementation:** `rust_optimizer/src/algorithms/rim_pressure.rs`

### 7. **CMA-ES (Covariance Matrix Adaptation)**
Advanced evolutionary strategy for continuous optimization:
- Adapts the search distribution based on successful mutations
- Used primarily for geometric strategies like ZipSkew

**Implementation:** `rust_optimizer/src/algorithms/cmaes_optimizer.rs`

---

## 🚀 How to Run (Docker)

All tools, compilers, and dependencies are packaged inside Docker. No local Rust or Python installation is required.

### 1. Build the Docker Image
```bash
docker compose build
```

### 2. Start an Interactive Container Session
```bash
docker compose run --rm optimizer bash
```
Inside `/app`, all tools (`cargo`, `rustc`, `python3`) and pre-built binaries are ready to use.

> 💡 **Tip**: All commands below can be executed directly inside the container bash, or from your host by prefixing them with `docker compose run --rm optimizer <command>`.

---

### Running Optimizations (Inside Container)

#### Basic Usage
```bash
# Optimize a specific solution
./rust_optimizer/target/release/christmas_tree_optimizer \
  -i solutions/T20.csv \
  --strategy sa \
  --generations 10000
```

#### Simulated Annealing (Recommended)
```bash
# Hot SA - Broad exploration
./rust_optimizer/target/release/christmas_tree_optimizer \
  -i solutions/T50.csv \
  --strategy sa \
  --generations 30000 \
  --sa-init-temp 0.001 \
  --sa-final-temp 0.00001 \
  --sa-step-scale 0.1

# Cold SA - Fine-tuning
./rust_optimizer/target/release/christmas_tree_optimizer \
  -i solutions/T50.csv \
  --strategy sa \
  --generations 50000 \
  --sa-init-temp 0.00001 \
  --sa-final-temp 0.000000001 \
  --sa-step-scale 0.01
```

#### Genetic Algorithm
```bash
# Evolve from existing solution
./rust_optimizer/target/release/christmas_tree_optimizer \
  -i solutions/T100.csv \
  --strategy ga \
  --generations 500 \
  --pop-size 50

# Pure random exploration (ignores input)
./rust_optimizer/target/release/christmas_tree_optimizer \
  --target-n 100 \
  --strategy ga \
  --pure-random \
  --generations 1000
```

#### Other Strategies
```bash
# Fine-tuning
./rust_optimizer/target/release/christmas_tree_optimizer \
  -i solutions/T75.csv \
  --strategy finetune \
  --fine-tune-iters 5000

# Gravity compression
./rust_optimizer/target/release/christmas_tree_optimizer \
  -i solutions/T75.csv \
  --strategy gravity \
  --gravity-steps 500

# Deca (base construction)
./rust_optimizer/target/release/christmas_tree_optimizer \
  --target-n 50 \
  --strategy deca
```

### Batch Processing with Python

#### `batch_deca_sa.py` ⭐ Primary Batch Script
Runs a multi-stage optimization pipeline for a range of N values:

```bash
python batch_deca_sa.py
```

**Configuration:**
- Edit `START_N` and `END_N` in the script (default: 11-40)
- Pipeline stages:
  1. **Deca**: Create base solution from T10 blocks
  2. **SA Hot**: Broad exploration (30k generations)
  3. **SA Cold**: Fine refinement (20k generations)
  4. **SA Frozen**: Micro-adjustments (50k generations)

**Output:** Saves improved solutions to `solutions/T{N}-2.csv`

---

## 🎨 Visualization with Rust Visualizer (GUI)

The Rust visualizer provides an interactive GUI for exploring and optimizing solutions in real-time. It runs through Docker using X11 socket forwarding on Linux.

### 1. Allow X11 Connections (Run once on host machine):
```bash
xhost +local:root
```

### 2. Launch Visualizer (From Host):
```bash
docker compose run --rm optimizer ./rust_optimizer/target/release/visualizer solutions/T25.csv
```
*(Or run `./rust_optimizer/target/release/visualizer solutions/T25.csv` directly inside the container bash).*

### Keyboard Controls

| Key | Action |
|-----|--------|
| **L** | Load solution (type filename like "T50") |
| **Left Arrow** | Navigate to previous solution (T_N-1) |
| **Right Arrow** | Navigate to next solution (T_N+1) |
| **V** | Toggle auto-reload (for live optimization) |
| **P** | Save current solution |
| **A** | Start Simulated Annealing (background) |
| **C** | Run CMA-ES optimization |
| **O** | Fine-tuning (100 iterations) |
| **G** | Apply gravity compression |
| **R** | Apply rim pressure |
| **F** | Explode (expand slightly to separate trees) |
| **X** | Shake (random perturbation) |
| **K** | Run Kaggle collision checker (Python) |
| **Q/E** | Rotate selected tree (hold Shift for fine control) |
| **Ctrl+Z** | Undo last change |

### Mouse Controls
- **Left Click + Drag**: Move a tree
- **Right Click + Drag**: Pan camera
- **Mouse Wheel**: Zoom in/out

### UI Panels
- **SA Configuration**: Adjust annealing parameters (temperature, generations, step scale, freeze zones)
- **Solution Navigator**: Quick navigation between solutions with Next/Back buttons

### Advanced: Live Optimization View
1. Start the visualizer: `./target/release/visualizer ../solutions/T50.csv`
2. Press **V** to enable auto-reload
3. Press **A** to start background SA optimization
4. Watch the solution evolve in real-time!

---

## 📊 Workflow Examples

### Creating a New Solution from Scratch
```bash
# 1. Create base with Deca
./rust_optimizer/target/release/christmas_tree_optimizer \
  --target-n 50 --strategy deca -o solutions/T50.csv

# 2. Hot SA exploration
./rust_optimizer/target/release/christmas_tree_optimizer \
  -i solutions/T50.csv --strategy sa --generations 30000 \
  --sa-init-temp 0.001 --sa-final-temp 0.00001

# 3. Cold SA refinement
./rust_optimizer/target/release/christmas_tree_optimizer \
  -i solutions/T50.csv --strategy sa --generations 50000 \
  --sa-init-temp 0.00001 --sa-final-temp 0.000000001
```

### Incremental Building (N → N+1)
```bash
# Use T49 as base for T50
./rust_optimizer/target/release/christmas_tree_optimizer \
  -i solutions/T49.csv -o solutions/T50.csv \
  --target-n 50 --strategy incremental
```

### Pruning (N → N-1)
```bash
# Remove one tree from T50 to create T49
./rust_optimizer/target/release/christmas_tree_optimizer \
  -i solutions/T50.csv -o solutions/T49.csv \
  --target-n 49 --strategy pruning
```

---

## 📁 Project Structure

```
ChrismasTree/
├── rust_optimizer/           # Main Rust implementation
│   ├── src/
│   │   ├── algorithms/       # Optimization algorithms
│   │   │   ├── annealing2.rs    # Simulated Annealing
│   │   │   ├── genetic.rs       # Genetic Algorithm
│   │   │   ├── deca.rs          # Deca strategy
│   │   │   ├── fine_tuning.rs   # Fine-tuning
│   │   │   ├── gravity.rs       # Gravity compression
│   │   │   └── ...
│   │   ├── bin/
│   │   │   └── visualizer.rs    # Interactive visualizer
│   │   ├── entities/         # Data structures (Tree)
│   │   └── utils/            # I/O and geometry utilities
│   └── Cargo.toml
├── solutions/                # Optimized solutions (T1.csv - T200.csv)
├── batch_deca_sa.py         # ⭐ Main batch optimization script
├── create_submission.py     # Generate Kaggle submission file
├── check_kaggle_collisions.py # Validate solutions
└── PROJECT_SUMMARY.md       # This file
```

---

## 🎯 Key Parameters Reference

### Simulated Annealing
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--sa-init-temp` | Initial temperature | 0.001 (hot), 0.00001 (cold) |
| `--sa-final-temp` | Final temperature | 0.00001 (hot), 0.000000001 (cold) |
| `--sa-step-scale` | Perturbation magnitude | 0.1 (hot), 0.01 (cold) |
| `--generations` | Number of iterations | 10000-100000 |
| `--freeze-inner` | Freeze inner % of trees | 0.0-1.0 |
| `--freeze-outer` | Freeze outer % of trees | 0.0-1.0 |

### Genetic Algorithm
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--pop-size` | Population size | 50-200 |
| `--generations` | Number of generations | 500-2000 |
| `--pure-random` | Ignore input, start random | flag |

### Fine-Tuning
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--fine-tune-iters` | Number of iterations | 1000-10000 |

---

## 🔧 Utilities

### Check for Collisions
```bash
python check_kaggle_collisions.py solutions/T50.csv
```

### Create Submission File
```bash
python create_submission.py
```
Generates `submission.csv` from all solutions in `solutions/` directory.

---

## 💡 Best Practices

1. **Always start with Deca** for N ≥ 20 to get a good initial layout
2. **Use SA in stages**: Hot → Medium → Cold for best results
3. **Visualizer is your friend**: Use it to understand what's happening and manually fix issues
4. **Save variants**: Use `-1`, `-2` suffixes (e.g., `T50-1.csv`) to keep multiple attempts
5. **Monitor scores**: The optimizer only saves if it improves the existing solution
6. **Batch processing**: Use `batch_deca_sa.py` for overnight optimization runs

---

## 🏅 Results

The optimization pipeline has successfully generated competitive solutions for N=1 to N=200, with the best results typically coming from:
- **Deca + Multi-stage SA** for large N (50-200)
- **GA + SA** for medium N (20-50)
- **Manual + Fine-tuning** for small N (1-20)

---

## 📝 Notes

- **Collision Detection**: Uses Kaggle-compatible logic (area-based intersection, not just touching)
- **Score Calculation**: Bounding square area / N (lower is better)
- **Auto-save**: Only saves if the new score is better than existing
- **Parallelization**: Most algorithms use Rayon for multi-threading

---

**Happy Optimizing! 🎄✨**
