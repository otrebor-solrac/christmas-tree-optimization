# Mathematical, Logical, and Algorithmic Foundations of Christmas Tree Packing

This document presents a comprehensive theoretical analysis of the Christmas Tree Packing problem, the mathematical formulation of its objective and constraints, the statistical mechanics of Simulated Annealing, and the geometric logic underpinning each optimization strategy.

---

## 1. Mathematical Problem Formulation

### 1.1 Geometry and Transformation Group
The object to be packed is a fixed, non-convex polygon $\mathcal{P} \subset \mathbb{R}^2$ with 15 coplanar vertices:

$$V = \{v_1, v_2, \dots, v_{15}\} \subset \mathbb{R}^2$$

The polygon consists of a central trunk and three tiered pairs of triangular branches, creating non-convex indentations and overhangs.

Every tree $i \in \{1, \dots, N\}$ in a configuration is an instance of $\mathcal{P}$ transformed by the Special Euclidean group $\text{SE}(2)$, defined by a translation vector $\mathbf{t}_i = (x_i, y_i) \in \mathbb{R}^2$ and an orientation angle $\theta_i \in [0, 2\pi)$:

$$\mathcal{P}_i(\mathbf{t}_i, \theta_i) = \mathbf{R}(\theta_i)\mathcal{P} + \mathbf{t}_i$$

Where the rotation matrix $\mathbf{R}(\theta) \in \text{SO}(2)$ is given by:

$$\mathbf{R}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

A complete arrangement of $N$ trees is parameterized by a state vector in configuration space $\mathbb{R}^{3N}$:

$$S = \big((x_1, y_1, \theta_1), (x_2, y_2, \theta_2), \dots, (x_N, y_N, \theta_N)\big) \in \mathbb{R}^{3N}$$

---

### 1.2 Objective Functional
Let $\mathcal{U}(S) = \bigcup_{i=1}^N \mathcal{P}_i$ be the union of all polygons in configuration $S$. The bounding box of the arrangement is determined by the coordinate extrema:

$$x_{\min}(S) = \inf_{(x,y) \in \mathcal{U}(S)} x, \quad x_{\max}(S) = \sup_{(x,y) \in \mathcal{U}(S)} x$$
$$y_{\min}(S) = \inf_{(x,y) \in \mathcal{U}(S)} y, \quad y_{\max}(S) = \sup_{(x,y) \in \mathcal{U}(S)} y$$

The spatial extents along the Cartesian axes are:

$$\Delta x(S) = x_{\max}(S) - x_{\min}(S), \quad \Delta y(S) = y_{\max}(S) - y_{\min}(S)$$

The enclosing square must accommodate the larger of the two spans, yielding a side length $L(S)$:

$$L(S) = \max\big(\Delta x(S), \Delta y(S)\big)$$

The normalized objective value (the official competition metric) is the area of the minimum bounding square divided by the number of trees $N$:

$$f(S) = \frac{L(S)^2}{N} = \frac{\big(\max(\Delta x(S), \Delta y(S))\big)^2}{N}$$

The global optimization problem is defined as:

$$\min_{S \in \mathcal{C}_{\text{free}}} f(S)$$

---

### 1.3 Feasible Configuration Space and Hard Collision Constraints
Let $\text{Int}(\mathcal{A})$ denote the interior of a planar set $\mathcal{A}$, and let $\mu(\cdot)$ denote the two-dimensional Lebesgue measure (area). Two trees $\mathcal{P}_i$ and $\mathcal{P}_j$ are non-overlapping if and only if their shared interior area is zero:

$$\mu\big(\mathcal{P}_i \cap \mathcal{P}_j\big) \le \epsilon \quad (\text{with } \epsilon = 10^{-12})$$

- **Boundary contact** ($\partial \mathcal{P}_i \cap \partial \mathcal{P}_j \ne \emptyset$ with zero overlap area) is valid and optimal for dense packing.
- **Interior intersection** ($\mu(\mathcal{P}_i \cap \mathcal{P}_j) > 10^{-12}$) violates the feasibility condition, penalizing the configuration with infinite cost:

$$\text{Energy}(S) = \begin{cases} f(S) & \text{if } \mu(\mathcal{P}_i \cap \mathcal{P}_j) \le \epsilon \quad \forall i \ne j \\ +\infty & \text{otherwise} \end{cases}$$

---

### 1.4 Search Space Complexity
The configuration space $\mathcal{C} = \mathbb{R}^{3N}$ exhibits extreme topological complexity:
1. **High Dimensionality**: For $N = 200$, the state space has 600 continuous dimensions.
2. **Severe Non-Convexity**: The presence of interlocking branches generates astronomical numbers of narrow local minima.
3. **Discontinuous Feasibility Boundaries**: Small rotations can cause sudden transition from valid contact to deep polygon overlap, creating steep energy barriers.

---

## 2. Theoretical Foundations of Simulated Annealing

Simulated Annealing models the optimization process as a thermodynamic system undergoing thermal equilibration according to the Boltzmann distribution.

### 2.1 Statistical Mechanics Analogy
In physical annealing, atoms in a molten substance possess high thermal mobility at elevated temperatures. As temperature decreases gradually, thermal fluctuations subside and the system condenses into a crystalline ground state characterized by minimal potential energy.

Under the canonical ensemble at temperature $T$, the probability distribution over configuration states $S$ is governed by the Gibbs-Boltzmann distribution:

$$P(S) = \frac{1}{Z(T)} \exp\left(-\frac{E(S)}{k_B T}\right)$$

Where $E(S)$ is the configuration energy, $k_B$ is the Boltzmann constant (conventionally set to 1 in numerical optimization), and $Z(T)$ is the partition function:

$$Z(T) = \int_{\mathcal{C}} \exp\left(-\frac{E(S)}{T}\right) dS$$

As $T \to 0^+$, the probability measure concentrates entirely on the set of global minima:

$$\lim_{T \to 0^+} P(S) = \begin{cases} \frac{1}{|\mathcal{S}^*|} & \text{if } S \in \mathcal{S}^* = \arg\min E(S) \\ 0 & \text{otherwise} \end{cases}$$

---

### 2.2 The Metropolis-Hastings Acceptance Criterion
Given a current state $S$ and a proposal state $S'$ generated by a symmetric neighborhood transition operator, the energy differential is:

$$\Delta E = E(S') - E(S)$$

The Metropolis acceptance probability $\alpha(S \to S')$ is defined as:

$$\alpha(S \to S') = \begin{cases} 1 & \text{if } \Delta E \le 0 \\ \exp\left(-\frac{\Delta E}{T}\right) & \text{if } \Delta E > 0 \end{cases}$$

This satisfies the **detailed balance condition** of the underlying Markov chain:

$$P(S) \cdot \alpha(S \to S') = P(S') \cdot \alpha(S' \to S)$$

Ensuring the stationary distribution of accepted states converges to the Boltzmann distribution at temperature $T$.

---

### 2.3 Non-Linear Annealing Schedule
The cooling schedule governs the rate at which temperature $T$ decays across iteration index $k \in [0, K_{\max}]$. To prevent premature freezing while maintaining convergence speed, a non-linear power cooling function is utilized:

$$T(k) = T_{\text{init}} \cdot \left(\frac{T_{\text{final}}}{T_{\text{init}}}\right)^{\left(\frac{k}{K_{\max}}\right)^p}$$

- **Power Parameter $p = 0.3$**: Compresses the extreme high-temperature regime and prolongs the critical transition interval where optimal clusters and orientations crystallize.
- **Dynamic Step Scaling**: The perturbation scale $\sigma_k$ contracts proportionally with the temperature:

$$\sigma(k) = \sigma_{\max} \cdot \left(\frac{T(k)}{T_{\text{init}}}\right)^\beta$$

Ensuring coarse global exploration when $T$ is high, transitioning seamlessly into microscopic boundary refinement as $T \to T_{\text{final}}$.

---

### 2.4 State Space Perturbation Logic

The proposal mechanism produces candidate transitions using three distinct operators:

#### 1. Continuous Spatial and Rotational Diffusion
A tree index $i$ is randomly selected, and its state is perturbed:

$$x_i' = x_i + \delta x, \quad y_i' = y_i + \delta y, \quad \theta_i' = \theta_i + \delta \theta$$

Where:
$$\delta x, \delta y \sim \mathcal{U}(-\text{spread}, \text{spread}), \quad \delta \theta \sim \mathcal{U}(-\theta_{\max}, \theta_{\max})$$
$$\text{spread} = \text{extent} \cdot 0.1 \cdot \sigma_k$$

#### 2. Pair Coordinate Transposition (Swap Mutation)
In tightly packed configurations, moving a single tree out of a locked cluster is mathematically infeasible without extensive temporary overlap. The swap operator exchanges the translation coordinates of two distinct trees $i$ and $j$ while preserving their respective orientations:

$$\mathbf{t}_i' = \mathbf{t}_j, \quad \mathbf{t}_j' = \mathbf{t}_i, \quad \theta_i' = \theta_i, \quad \theta_j' = \theta_j$$

This discrete jump allows internal and external trees to trade topological positions instantly across the energy landscape.

#### 3. Onion Peeling Partitioning (Radial Freeze Zones)
Trees are classified by Euclidean distance from the layout center of mass $\mathbf{c} = \frac{1}{N}\sum_{i=1}^N \mathbf{t}_i$:

$$d_i = \|\mathbf{t}_i - \mathbf{c}\|_2$$

- **Inner Freeze Zone ($\rho_{\text{inner}}$)**: Trees within percentile threshold $\rho_{\text{inner}}$ are locked in place. Mutations are restricted exclusively to the peripheral shell, preventing disruption of an already optimized core.
- **Outer Freeze Zone ($\rho_{\text{outer}}$)**: The external perimeter is locked, allowing annealing to resolve vacancies in the interior.

---

## 3. Algorithmic and Logical Catalog of Strategies

The system implements multiple independent and composable strategies targeting different structural scales:

```
                          ┌───────────────────────────┐
                          │   Initial Problem State   │
                          └─────────────┬─────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
   Modular Construction       Continuous Optimization       Evolutionary Search
   ├── Deca Tiling            ├── Simulated Annealing       └── Genetic Algorithm
   ├── Mosaic 2x2             └── ZipSkew (CMA-ES)              (Coupled Pairs)
   └── Grid Zipper
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        │
                                        ▼
                             Boundary Compaction
                             ├── Gravity Centroid Pull
                             └── Rim Pressure Field
                                        │
                                        ▼
                             Microscopic Polishing
                             └── Deterministic Fine-Tuning
```

---

### 3.1 Deca Strategy (Modular Crystalline Tiling)
* **Target Domain**: Large instances ($N \ge 20$).
* **Mathematical Concept**: Modular reduction of degrees of freedom.
* **Mechanism**: 
  Instead of solving for $3N$ independent variables simultaneously, the Deca strategy uses an analytically optimized 10-tree fundamental unit cell ($T_{10}$). These cells interlock with maximal planar density.
  
  The layout is generated by evaluating optimal translations of the unit cell on a 2D Bravais lattice:
  
  $$\mathbf{R}_{u,v} = u \mathbf{a}_1 + v \mathbf{a}_2$$
  
  Where $\mathbf{a}_1, \mathbf{a}_2 \in \mathbb{R}^2$ are lattice basis vectors. Excess trees beyond $N$ are removed using margin pruning. This provides a dense, collision-free starting seed far superior to random placement.

---

### 3.2 ZipSkew Strategy (Parametric Lattice Optimization via CMA-ES)
* **Target Domain**: Intermediate and large instances ($N \in [20, 200]$).
* **Mathematical Concept**: Continuous evolutionary optimization of lattice parameters.
* **Mechanism**:
  Trees pack with high efficiency when arranged in alternating interlocking rows with opposite parity ($\theta \in \{0, \pi\}$). The geometry of the entire layout is parameterized by a continuous vector $\mathbf{p} \in \mathbb{R}^6$:
  
  $$\mathbf{p} = (\text{stride}_x, \text{row}_y, \text{skew}_x, \text{gap}_x, \text{offset}_y, \text{margin})$$
  
  The Covariance Matrix Adaptation Evolution Strategy (CMA-ES) samples candidate parameter vectors from a multivariate normal distribution:
  
  $$\mathbf{p}^{(g+1)} \sim \mathbf{m}^{(g)} + \sigma^{(g)} \mathcal{N}(\mathbf{0}, \mathbf{C}^{(g)})$$
  
  The mean vector $\mathbf{m}$, step-size $\sigma$, and covariance matrix $\mathbf{C}$ update iteratively based on the rank of resulting bounding square areas, learning the correlation between spatial parameters and optimal packing density.

---

### 3.3 Genetic Algorithm (GA) with Coupled Pair Dynamics
* **Target Domain**: Global exploration from scratch or escaping deep local minima.
* **Mathematical Concept**: Evolutionary population dynamics with domain-specific crossover.
* **Mechanism**:
  - **Representation**: An individual is represented as a state vector $S = (\mathbf{t}_1, \theta_1, \dots, \mathbf{t}_N, \theta_N)$.
  - **Tournament Selection**: Fitter individuals with smaller bounding box scores and zero overlap are chosen for reproduction.
  - **Spatial Crossover**: A random cutting plane $\mathbf{n} \cdot \mathbf{x} = d$ partitions the layout. Offspring inherit trees on one side of the plane from Parent A, and trees on the opposite side from Parent B, followed by collision repair.
  - **Coupled Pair Mutation (`groups`)**: Trees that form complementary pairs (facing tip-to-trunk) are grouped into rigid compound bodies. Mutation operators move and rotate both trees together, preserving their tight mutual contact while reorienting the pair within the global cluster.

---

### 3.4 Centroid Gravity Compression
* **Target Domain**: Post-processing intermediate layouts.
* **Mathematical Concept**: Artificial conservative potential field.
* **Mechanism**:
  Every tree is subjected to an attractive potential energy field centered at the geometric centroid $\mathbf{c} = \frac{1}{N}\sum_{i=1}^N \mathbf{t}_i$:
  
  $$U(\mathbf{t}_i) = \frac{1}{2} k \|\mathbf{t}_i - \mathbf{c}\|_2^2$$
  
  The corresponding inward force vector is:
  
  $$\mathbf{F}_i = -\nabla U(\mathbf{t}_i) = -k(\mathbf{t}_i - \mathbf{c})$$
  
  Trees iteratively step along their normalized force vectors $\hat{\mathbf{F}}_i$ by a discrete step $\eta$:
  
  $$\mathbf{t}_i^{(m+1)} = \mathbf{t}_i^{(m)} + \eta \frac{\mathbf{c} - \mathbf{t}_i^{(m)}}{\|\mathbf{c} - \mathbf{t}_i^{(m)}\|_2}$$
  
  If any displacement induces a polygon intersection with neighboring trees, the step is halted at the boundary. This draws external trees inward and compacts internal voids without altering orientations.

---

### 3.5 Rim Pressure Compression
* **Target Domain**: Squaring the perimeter envelope.
* **Mathematical Concept**: Inward normal projection from the convex hull boundary.
* **Mechanism**:
  1. Computes the convex hull $\mathcal{H} = \text{Conv}\big(\bigcup_{i=1}^N \mathcal{P}_i\big)$.
  2. Identifies boundary trees whose vertices intersect or form the edges of $\mathcal{H}$.
  3. Projects an inward displacement vector orthogonal to the adjacent hull facets.
  4. Advances perimeter trees inward while internal trees remain fixed, directly reducing the exterior bounding envelope.

---

### 3.6 Deterministic Fine-Tuning
* **Target Domain**: Terminal convergence.
* **Mathematical Concept**: Discrete coordinate descent with monotonic non-increasing energy.
* **Mechanism**:
  For each tree $i \in \{1, \dots, N\}$, the algorithm performs small test steps along orthogonal coordinate axes and angular increments:
  
  $$\Delta \in \{(\pm \epsilon_x, 0, 0), (0, \pm \epsilon_y, 0), (0, 0, \pm \epsilon_\theta)\}$$
  
  A displacement is accepted if and only if:
  1. No polygon collision occurs ($\mu(\mathcal{P}_i \cap \mathcal{P}_j) \le 10^{-12}$).
  2. The global bounding side strictly decreases: $L(S') < L(S)$.
  
  This deterministic sweep guarantees convergence to the local minimum of the active basin without stochastic regression.

---

### 3.7 Boundary Value Propagation: Pruning and Incremental Growth
* **Target Domain**: Inductive solution generation across adjacent problem sizes ($N-1 \leftrightarrow N \leftrightarrow N+1$).
* **Mechanism**:
  - **Pruning ($N+1 \to N$)**: Evaluates marginal contribution of each tree to the bounding square:
    
    $$\Delta L_i = L(S) - L(S \setminus \{\mathcal{P}_i\})$$
    
    The tree exhibiting the largest $\Delta L_i$ (the most spatially expensive perimeter tree) is removed, yielding an immediately compact and valid configuration for $N$.
  - **Incremental Growth ($N-1 \to N$)**: Identifies perimeter cavities in the valid solution $S_{N-1}$. A new tree is inserted at the boundary point that minimizes the expansion of $\max(\Delta x, \Delta y)$, followed by local Simulated Annealing.

---

## 4. Collision Detection and Polygon Verification Mathematics

### 4.1 Hierarchical Spatial Filtering
Direct pairwise polygon intersection testing for $N=200$ requires $\binom{200}{2} = 19,900$ comparisons per iteration. To achieve millions of evaluations per second, a two-tier spatial pruning pipeline is utilized:

1. **Broad Phase (Axis-Aligned Bounding Box)**:
   For each polygon $\mathcal{P}_i$, the interval extents $[x_i^{\min}, x_i^{\max}]$ and $[y_i^{\min}, y_i^{\max}]$ are precalculated.
   Polygons can intersect only if their bounding boxes overlap on both projection axes:
   
   $$\big[x_i^{\min}, x_i^{\max}\big] \cap \big[x_j^{\min}, x_j^{\max}\big] \ne \emptyset \quad \land \quad \big[y_i^{\min}, y_i^{\max}\big] \cap \big[y_j^{\min}, y_j^{\max}\big] \ne \emptyset$$
   
   This rejection test runs in $O(1)$ scalar arithmetic and eliminates over 95% of candidate pairs.

2. **Narrow Phase (Exact Polygon Clipping)**:
   Candidate pairs passing the broad phase are evaluated using polygon clipping algorithms (Vatti / Sutherland-Hodgman) to compute the exact intersection polygon:
   
   $$\mathcal{I}_{ij} = \mathcal{P}_i \cap \mathcal{P}_j$$
   
   Green's theorem determines the area of the resulting intersection polygon:
   
   $$\mu(\mathcal{I}_{ij}) = \frac{1}{2} \left| \sum_{k=1}^{M} (x_k y_{k+1} - x_{k+1} y_k) \right|$$
   
   If $\mu(\mathcal{I}_{ij}) > 10^{-12}$, collision is declared and the mutation is rejected.

---

## 5. Thread Decoupling in the Real-Time Visualizer

The interactive visualizer implements asynchronous multi-threading to maintain continuous graphic responsiveness during intensive optimization:

1. **Background Computation Worker**:
   - Executes the Simulated Annealing state transitions continuously at hundreds of thousands of evaluations per second.
   - At fixed periodic intervals (every 2,000 iterations), flushes a serialized snapshot of the active candidate state to shared disk storage (`live.csv`).

2. **Foreground Graphical Thread (60 FPS)**:
   - Polls the shared state file asynchronously at 10 Hz without blocking rendering operations.
   - Computes dynamic camera projection matrices, mapping arbitrary world coordinates to the normalized device coordinate viewport:
     
     $$\mathbf{x}_{\text{screen}} = \mathbf{M}_{\text{viewport}} \mathbf{M}_{\text{camera}} \mathbf{x}_{\text{world}}$$
     
   - Renders polygon meshes, boundary contours, and real-time metric HUDs while the background worker continues optimization uninterrupted.
