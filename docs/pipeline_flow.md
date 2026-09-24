# Optimization Pipeline Flow

All commands are designed to be run inside the Docker container (`docker compose run --rm optimizer bash`) or prefixed with `docker compose run --rm optimizer`.

## Input Command

```bash
python3 run_rust.py --target N [--strategy S] [--optimizer O] [--gen G] [--fine-tune F]
```

### Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--target` | required | 1-200 | Target number of trees |
| `--strategy` | `all` | `all`, `mosaic`, `zipper`, `pruning`, `incremental`, `none` | Geometric strategy |
| `--optimizer` | `ga` | `ga`, `cmaes`, `sa` | Optimizer (if strategies fail) |
| `--gen` | 10 | integer | Optimizer generations / iterations |
| `--fine-tune` | 0 | integer | Fine-tuning iterations |
| `--steps` | 50 | integer | Gravity compression steps |
| `--pop` | 100 | integer | Population size (GA only) |

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                INPUT                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  python3 run_rust.py --target N [--strategy S] [--optimizer O]          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: GEOMETRIC STRATEGIES                        │
│                    (If --strategy != "none")                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─ Mosaic ──────────────────────────┐                                  │
│  │ • Only if N % 4 == 0              │                                  │
│  │ • Uses T{N/4}.csv as seed         │                                  │
│  │ • Creates 2x2 tiled mosaic        │                                  │
│  └───────────────────────────────────┘                                  │
│                                                                         │
│  ┌─ Grid Zipper ─────────────────────┐                                  │
│  │ • Generates from scratch          │                                  │
│  │ • Zigzag pattern with             │                                  │
│  │   configurable stride_x & row_h   │                                  │
│  └───────────────────────────────────┘                                  │
│                                                                         │
│  ┌─ Pruning ─────────────────────────┐                                  │
│  │ • Reads T{N+1}.csv                │                                  │
│  │ • Removes 1 tree (worst impact)   │                                  │
│  └───────────────────────────────────┘                                  │
│                                                                         │
│  ┌─ Incremental ─────────────────────┐                                  │
│  │ • Reads T{N-1}.csv                │                                  │
│  │ • Adds 1 tree on the perimeter    │                                  │
│  └───────────────────────────────────┘                                  │
│                                                                         │
│  → Result: Best strategy with finite score                              │
│            Or none if all fail                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            Any succeeded?                     All failed?
                    │                               │
                    ▼                               ▼
┌──────────────────────────┐     ┌─────────────────────────────────────────┐
│ Use best strategy        │     │         PHASE 2: OPTIMIZER              │
│ as candidate             │     │         (Only if phase 1 failed)        │
└──────────────────────────┘     ├─────────────────────────────────────────┤
            │                    │                                         │
            │                    │  Loads existing T{N}.csv or generates   │
            │                    │  seed using Grid Zipper                 │
            │                    │                                         │
            │                    │  According to --optimizer:              │
            │                    │  ┌─ ga ──────────────────────┐          │
            │                    │  │ Genetic Algorithm         │          │
            │                    │  │ • Population + Selection  │          │
            │                    │  │ • Crossover + Mutation    │          │
            │                    │  └───────────────────────────┘          │
            │                    │                                         │
            │                    │  ┌─ cmaes ───────────────────┐          │
            │                    │  │ Adaptive CMA-ES           │          │
            │                    │  │ • 4 restarts with growing │          │
            │                    │  │   sigma (5% → 50%)        │          │
            │                    │  └───────────────────────────┘          │
            │                    │                                         │
            │                    │  ┌─ sa ──────────────────────┐          │
            │                    │  │ Simulated Annealing       │          │
            │                    │  │ • Accepts worse solutions │          │
            │                    │  │ • Decreasing temperature  │          │
            │                    │  └───────────────────────────┘          │
            │                    └─────────────────────────────────────────┘
            │                               │
            └───────────┬───────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: REFINEMENT                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. save_if_better() ─────────────────────────────────────────────────── │
│     • Checks whether candidate has collisions                           │
│     • Checks whether existing file has collisions                       │
│     • Saves if: new is valid AND (old invalid OR new better score)      │
│                                                                         │
│  2. Gravity (--steps steps) ─────────────────────────────────────────── │
│     • Moves trees toward the center                                     │
│     • Compacts layout                                                   │
│     • save_if_better()                                                  │
│                                                                         │
│  3. Fine-Tuning (--fine-tune iters) ──────────────────────────────────── │
│     • Small random perturbations                                        │
│     • Accepts improvements only                                         │
│     • save_if_better()                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                OUTPUT                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  solutions/T{N}.csv                                                     │
│  - Best solution found (or existing one if no improvement)              │
│  - Format: id, x, y, deg, score                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Dependencies

```
T1.csv  ←───────────────────────────────────────────────────────────────┐
T2.csv  ← Incremental(T1)                                               │
T3.csv  ← Incremental(T2)                                               │
T4.csv  ← Mosaic(T1) or Incremental(T3)                                  │
...                                                                     │
T50.csv ← Zipper, Pruning(T51), Incremental(T49)                        │
...                                                                     │
T200.csv ← Zipper, Pruning(T201), Mosaic(T50)                           │
                                                                        │
                    Pruning reads N+1 file ─────────────────────────────┘
```

---

## Strategy Details

### Mosaic
- **Condition**: N is a multiple of 4
- **Input**: `solutions/T{N/4}.csv`
- **Process**: Duplicates base solution into a 2x2 grid, scaling coordinates

### Grid Zipper
- **Condition**: Always available
- **Input**: None (generates from scratch)
- **Process**: Creates alternating zigzag rows (0° and 180° rotations)
- **Parameters**: `stride_x` (horizontal spacing), `row_height` (vertical spacing)

### Pruning
- **Condition**: `T{N+1}.csv` exists
- **Input**: `solutions/T{N+1}.csv`
- **Process**: Removes the tree with the least negative impact on the bounding box

### Incremental
- **Condition**: `T{N-1}.csv` exists
- **Input**: `solutions/T{N-1}.csv`
- **Process**: Adds a new tree along the perimeter of the existing layout

---

## Optimizer Details

### GA (Genetic Algorithm)
- **Use**: Default when geometric strategies fail
- **Process**: 
  - Generates candidate population
  - Tournament selection
  - Crossover between elite individuals
  - Random mutation

### CMA-ES (Covariance Matrix Adaptation)
- **Use**: `--optimizer cmaes`
- **Process**:
  - Continuous optimization based on Gaussian distribution
  - 4 restarts with expanding sigma (5%, 15%, 30%, 50%)
  - Escapes local minima via adaptive covariance search

### SA (Simulated Annealing)
- **Use**: `--optimizer sa`
- **Process**:
  - Probabilistically accepts worse solutions to escape local traps
  - High initial temperature → cooling down
  - High exploratory capability

---

## Refinement Details

### save_if_better() Logic
```
IF new_has_collisions:
    DO NOT SAVE
    
IF file_already_exists:
    IF existing_file_has_collisions:
        SAVE (replaces invalid solution)
    ELSE IF new_score < old_score:
        SAVE (verified improvement)
ELSE:
    SAVE (new solution created)
```

### Gravity Compression
- Drags each tree toward the layout centroid
- Small step increment (0.01 per iteration)
- Performs collision checks after each adjustment

### Fine-Tuning
- Micro perturbations on x, y, and angle
- Accepts only changes that strictly reduce score
- Fast local polish for boundary tightening

---

## Usage Examples (Inside Docker)

Execute these inside the container (`docker compose run --rm optimizer bash`):

```bash
# Run all strategies for T50
python3 run_rust.py --target 50 --fine-tune 100

# Run pruning only for T50
python3 run_rust.py --target 50 --strategy pruning --fine-tune 100

# Force CMA-ES without geometric seeds
python3 run_rust.py --target 50 --strategy none --optimizer cmaes --gen 100

# Simulated Annealing with many iterations
python3 run_rust.py --target 50 --strategy none --optimizer sa --gen 50

# Pruning cascade from 200 down to 1
for n in {200..1}; do python3 run_rust.py --target $n --strategy pruning --fine-tune 100; done
```
