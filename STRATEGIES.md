# 🎄 Christmas Tree Packing Strategies (v2.0)

This document details the algorithms and strategies currently deployed in the Christmas Tree Packing optimization project. It has been updated to reflect the transition to a hybrid Python/Rust architecture and the introduction of geometric strategies like ZipSkew.

## 🎯 Global Objective

Minimize the area of the convex hull enclosing a set of $N$ non-overlapping Christmas trees.

$$
\text{Score} = \frac{\text{Area}(\text{ConvexHull}(\{T_1, T_2, ..., T_N\}))}{N}
$$

---

## 🏗️ Architecture Overview

The system uses a **Hybrid approach**:
1.  **Python (`app/algorithms/zip_skew_batch.py`)**: Responsible for finding optimal geometric structures (Lattices) using **CMA-ES**. Python's `shapely` library is currently more robust for these complex union operations.
2.  **Rust (`rust_optimizer/`)**: Responsible for high-performance refinement, collision detection, and large-scale random exploration (Genetic Algorithm).

---

## 📐 Geometric Strategies

### 1. ZipSkew Algorithm (`zip_skew.py` / `zipskew` strategy)
**Best for:** $N > 20$, massive layouts.

Based on the observation that trees pack efficiently in skewed, zipper-like rows.
*   **Concept**: Parameters define a "Skewed Lattice" (row height, skew factor, column step).
*   **Optimization**: Uses **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) to find the 6 float parameters that generate the tightest non-overlapping lattice for $N$ trees.
*   **Status**: 
    *   **Python**: **PRIMARY**. Very effective. Can batch process multiple $N$.
    *   **Rust**: Implemented using `cmaes` crate but currently less effective than Python at converging to the global optimum. Halted by default (can be enabled via flag).

### 2. Grid Zipper (`zipper` strategy)
**Best for:** Seeding.

Generates a deterministic grid-based layout. Simple but provides a valid starting point for refinement.

---

## � Genetic Algorithm (Rust)

**Mode: `--strategy ga` or `--optimizer ga`**

A high-performance GA written in Rust to explore the solution space or refine existing solutions.

### Key Features:
*   **Population**: 50-200 individuals (vectors of Tree objects).
*   **Initialization**:
    *   **Hybrid**: 80% Random / 20% Variations of input.
    *   **Pure Random (`--pure-random`)**: 100% Random layouts. Useful for finding completely new configurations when deterministic strategies get stuck.
*   **Cost Function**:
    *   Valid: $\text{Area}/N$
    *   Collision: $\infty$ (Infinite penalty forces validity).
*   **Operators**:
    *   **Gravity Refinement**: Pulls trees towards the center to compact the layout.
    *   **Fine-Tuning**: Micro-rotations and shifts to optimize finding the minimum bounding box.
    *   **Earthquake**: Resets part of the population if stagnation is detected (25 generations without improvement).

---

## 🚀 Current Workflow (Inside Docker)

All commands are executed inside the Docker container (`docker compose run --rm optimizer bash`):

1.  **Step 1: Geometric Search (Python)**
    Run `zip_skew_batch.py` to find the best "shape" for $N$.
    ```bash
    python app/algorithms/zip_skew_batch.py --n 197
    ```
    *Output*: Saves to `solutions/T197.csv`.

2.  **Step 2: Refinement & Validation (Rust)**
    Run the Rust optimizer to apply gravity, fine-tuning, and ensure validity.
    ```bash
    ./rust_optimizer/target/release/christmas_tree_optimizer -i solutions/T197.csv -o solutions/T197.csv --target-n 197 --strategy all
    ```

3.  **Step 3: Random Exploration (Optional)**
    If geometric strategies fail, use the Rust GA in pure random mode to brute-force a new configuration.
    ```bash
    ./rust_optimizer/target/release/christmas_tree_optimizer -i solutions/T197.csv -o solutions/T197.csv --target-n 197 --pure-random --generations 1000
    ```

---

## ⚠️ Challenges & Next Steps

### 1. CMA-ES Parity (Python vs Rust)
The Rust implementation of ZipSkew (CMA-ES) struggles to find the same quality of minima as the Python `cma` library.
*   *Theory*: The objective function in Rust might handle collisions too harshly (returning Infinity), preventing the optimizer from traversing "infeasible" regions to find better feasible ones. Python uses a soft penalty ($100 + \text{overlap} \times 10^5$).
*   *Action*: Implement soft collision penalties in Rust's ZipSkew objective function.

### 2. Genetic Algorithm Efficiency
For large $N$ (e.g., $N=197$), the GA is slow because `gravity_steps` (20-30) are applied to every individual in every generation.
*   *Action*: Reduce `gravity_steps` during evolution (e.g., to 5) and only do full compaction (30+) for the best individual or at the end.

### 3. Multi-Strategy Pipeline
Currently, we manually switch between Python and Rust.
*   *Action*: Create a master script that runs Python ZipSkew, then immediately triggers Rust refinement, then loops.

### 4. Empty File Handling
We recently fixed a bug where Rust panicked on empty files with `--pure-random`. It now auto-seeds linear trees.
