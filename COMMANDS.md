# 🎄 Optimizer Usage Guide (Docker & Rust)

This guide explains how to use the Rust optimizer (`christmas_tree_optimizer`) inside Docker to solve and refine the Christmas Tree Packing problem.

All dependencies, compilers, and tools are containerized. You do not need to install Rust or Cargo on your host system.

---

## 🐳 Running inside Docker

### 1. Start an Interactive Container Shell
From the repository root on your host:
```bash
docker compose run --rm optimizer bash
```
You will enter `/app` inside the container where all commands can be run directly.

> 💡 **One-Off Execution from Host**: You can also run any command directly from your host terminal by prepending `docker compose run --rm optimizer`, for example:
> ```bash
> docker compose run --rm optimizer ./rust_optimizer/target/release/christmas_tree_optimizer --target-n 20 --strategy sa --generations 5000
> ```

---

## 🚀 Key Concepts

1. **Automatic I/O**:
   - If `-i` (input) is not specified, it attempts to load from the `-o` (output) file to resume progress.
   - If `-o` (output) is not specified, it defaults to `solutions/T[N].csv`.
2. **Kaggle Score**:
   - The program calculates the official score: $\text{Bounding Square Area} / N$.
   - **It only overwrites the output file if the new solution achieves a better score.**
3. **Automatic N Adjustment**:
   - If the input file has fewer trees than `--target-n`, missing trees are appended automatically.

---

## 🛠️ Strategy Commands

Run these inside the container terminal (at `/app`):

### 1. Genetic Algorithm (GA)
Ideal for broad global exploration and discovering new packing topologies.
```bash
# Evolve an existing solution (500 generations)
./rust_optimizer/target/release/christmas_tree_optimizer -i solutions/T197.csv --strategy ga --generations 500

# Pure random exploration from scratch (ignores input)
./rust_optimizer/target/release/christmas_tree_optimizer --target-n 6 --strategy ga --pure-random --generations 1000
```

### 2. Fine-Tuning
Performs micro-rotations and translations for precision boundary tightening.
```bash
./rust_optimizer/target/release/christmas_tree_optimizer -i solutions/T197.csv --strategy finetune --fine-tune-iters 5000
```

### 3. Gravity Compression
Pushes all trees toward the center $(0,0)$ while avoiding collisions.
```bash
./rust_optimizer/target/release/christmas_tree_optimizer -i solutions/T197.csv --strategy gravity --gravity-steps 500
```

### 4. Simulated Annealing (SA) ⭐ Recommended
Excellent for escaping local minima where greedy methods get stuck.
```bash
# Quick annealing run
./rust_optimizer/target/release/christmas_tree_optimizer -i solutions/T6.csv --strategy sa --generations 200

# Thorough multi-temperature SA
./rust_optimizer/target/release/christmas_tree_optimizer \
  -i solutions/T50.csv \
  --strategy sa \
  --generations 30000 \
  --sa-init-temp 0.001 \
  --sa-final-temp 0.00001 \
  --sa-step-scale 0.1
```

### 5. ZipSkew (Geometric Lattice)
Lattice-based packing strategy optimized with CMA-ES.
```bash
./rust_optimizer/target/release/christmas_tree_optimizer --target-n 197 --strategy zipskew --generations 100
```

### 6. Automated Pipeline (`all`)
Runs a predefined multi-stage pipeline (ZipSkew -> Repair -> Gravity -> FineTune).
```bash
./rust_optimizer/target/release/christmas_tree_optimizer --target-n 197 --strategy all
```

### 7. Pruning (Top-Down)
Removes trees intelligently to reach a smaller $N$.
```bash
# Prune from T197 to T196
./rust_optimizer/target/release/christmas_tree_optimizer -i solutions/T197.csv -o solutions/T196.csv --target-n 196 --strategy pruning

# Automatically search for T{N+1} to prune down to target N
./rust_optimizer/target/release/christmas_tree_optimizer --target-n 6 --strategy pruning
```

### 8. Incremental (Bottom-Up)
Adds trees to the outer perimeter of an existing solution.
```bash
# Expand T5 to T6
./rust_optimizer/target/release/christmas_tree_optimizer -i solutions/T5.csv -o solutions/T6.csv --target-n 6 --strategy incremental

# Automatically find T{N-1} to increment up to target N
./rust_optimizer/target/release/christmas_tree_optimizer --target-n 197 --strategy incremental
```

---

## 📈 Example Workflows

### Transitioning from $N=5$ to $N=6$ (Incremental)
Use a solid record in $T_5$ as the seed for $T_6$:
```bash
./rust_optimizer/target/release/christmas_tree_optimizer -i solutions/T5.csv -o solutions/T6.csv --target-n 6 --strategy ga
```

### Cascading Polish
Chain commands to progressively refine an arrangement:
1. **Step 1 (Base Construction)**: `./rust_optimizer/target/release/christmas_tree_optimizer --target-n 20 --strategy zipper`
2. **Step 2 (Compacting)**: `./rust_optimizer/target/release/christmas_tree_optimizer -i solutions/T20.csv --strategy gravity --gravity-steps 1000`
3. **Step 3 (Fine Polish)**: `./rust_optimizer/target/release/christmas_tree_optimizer -i solutions/T20.csv --strategy finetune --fine-tune-iters 10000`

---

## 📋 Common Parameters Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-i`, `--input` | Input solution file (`.csv`) | *(Optional)* |
| `-o`, `--output` | Destination file for improvements | `solutions/T[N].csv` |
| `--target-n` | Target number of trees | Inferred from filename |
| `--strategy` | Strategy name (`sa`, `ga`, `finetune`, `gravity`, `deca`, `pruning`, `incremental`, `all`) | `all` |
| `--generations` | Iterations for GA, SA, CMA-ES | `20` |
| `--pop-size` | Population size (GA only) | `50` |
| `--gravity-steps`| Compression steps for gravity | `20` |
| `--fine-tune-iters`| Iterations for fine-tuning | `1000` |
| `--pure-random` | Ignore input and start from random placements | `false` |
