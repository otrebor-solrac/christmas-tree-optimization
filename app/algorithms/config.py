"""
Configuration module for Simulated Annealing parameters.
"""
from dataclasses import dataclass

@dataclass
class AnnealingConfig:
    """
    Centralized configuration for the Simulated Annealing algorithm.
    
    This class contains all tunable parameters that control the behavior of the
    simulated annealing optimization process for packing Christmas trees.
    """
    
    # ==================== COOLING SCHEDULE ====================
    # Controls how quickly the algorithm "cools down" and becomes more selective
    
    alpha: float = 0.95
    """Cooling rate (geometric decay). Range: 0.9-0.99
    - Higher values (0.98): Slower cooling, more exploration, longer runtime
    - Lower values (0.90): Faster cooling, quicker convergence, may miss optimal solutions
    - Default 0.95 is a good balance for most cases"""
    
    min_temp: float = 0.005
    """Stopping temperature. When temp drops below this, annealing stops.
    - Lower values: More iterations, finer optimization
    - Higher values: Faster termination, coarser results"""
    
    # ==================== PENALTIES ====================
    # Control how strongly the algorithm avoids undesirable states
    
    initial_penalty_weight: float = 50.0
    """Base weight for soft overlap penalty. Increases during optimization.
    - Higher values: Stronger avoidance of overlaps early on
    - Lower values: More tolerance for temporary overlaps during exploration"""
    
    strict_collision_penalty: float = 100000.0
    """Penalty for hard collisions (actual tree intersections).
    - Very high value ensures collisions are strongly rejected
    - Should be much larger than other penalties"""
    
    compaction_base_penalty: float = 1000000.0
    """Penalty for exceeding the maximum allowed area during compaction mode.
    - Activates when trees are too spread out (ratio > 2.0)
    - Prevents area from growing once compaction starts"""
    
    # ==================== PERTURBATION ====================
    # Controls how trees are moved during optimization
    
    angle_noise_range: float = 15.0
    """Random rotation range in degrees (±).
    - Higher values (15-20): More angular exploration, helps escape local minima
    - Lower values (5): More stable, less chaotic rotations
    - Default 10° allows moderate rotation exploration"""
    
    # ==================== INSERTION STRATEGIES ====================
    # Different parameter sets for different insertion attempts
    
    # --- First Attempt: Fine Tuning ---
    attempt_1_temp: float = 10.0
    """Initial temperature for first insertion attempt.
    - Low temp = conservative, small adjustments
    - Uses the geometric strategy placement as starting point"""
    
    attempt_1_mag: float = 0.1
    """Movement magnitude factor for first attempt.
    - Small value = fine adjustments around initial placement"""
    
    attempt_1_iters: int = 5000
    """Iterations for first attempt.
    - Fewer iterations since we're starting from a good position"""
    
    # --- Retry Attempts: Global Search ---
    attempt_retry_temp: float = 10.0
    """Initial temperature for retry attempts (when first attempt fails).
    - High temp = very exploratory, accepts worse states more often
    - Helps escape from bad initial placements"""
    
    attempt_retry_mag: float = 3.0
    """Movement magnitude for retries.
    - Large value = big jumps, explores distant positions"""
    
    attempt_retry_iters: int = 5000
    """Iterations for retry attempts.
    - More iterations to thoroughly explore the space"""
    
    max_insertion_attempts: int = 100
    """Maximum number of attempts to insert a new tree before giving up.
    - Higher values: More persistent, slower but more likely to succeed
    - Lower values: Faster failure, may not find valid placements"""

    exhaustive_search: bool = True
    """If True, continues searching after finding a valid placement to find the BEST 
    one among max_insertion_attempts. If False, stops at first success."""
    
    # ==================== OPTIMIZATION OF EXISTING TREES ====================
    # Used when optimizing trees that are already placed (n_new = 0)
    
    mutable_opt_iters: int = 2000
    """Iterations when optimizing existing trees without adding new ones."""
    
    mutable_opt_temp: float = 5.0
    """Temperature for existing tree optimization.
    - Moderate value allows some exploration while refining positions"""
    
    mutable_opt_mag: float = 1.0
    """Movement magnitude for existing tree optimization.
    - Moderate value allows repositioning without excessive chaos"""

    # ==================== PACKING GEOMETRY ====================
    # Grid spacing parameters for the square packing strategy
    
    stride_x: float = 0.42
    """Horizontal spacing between tree columns in the grid.
    - Smaller values: Tighter packing, more overlap risk
    - Larger values: Looser packing, larger bounding box
    - 0.42 is optimized for tree width with zipper pattern"""
    
    row_height: float = 0.85
    """Vertical spacing between tree rows.
    - Smaller values: Tighter vertical packing
    - Larger values: More vertical space
    - 0.85 allows for vertical nesting with alternating orientations"""
    
    # ==================== VISUALIZATION ====================
    # Controls visualization behavior during optimization
    
    viz_frames: int = 20
    """Number of visualization frames to show during annealing.
    - Higher values: More frequent updates, slower but more detailed
    - Lower values: Fewer updates, faster
    - 0: No visualization updates during optimization"""

@dataclass
class GravityConfig:
    """
    Configuration for Gravity/Center of Mass Compaction Algorithm.
    """
    steps: int = 1000
    step_size: float = 0.01
    noise_range: float = 0.8  # Radians
    teleport_interval: int = 50
    enable_teleport: bool = True
    visualize: bool = False
    viz_step: int = 20

@dataclass
class GeneticConfig:
    """Configuration for Genetic Algorithm."""
    population_size: int = 50
    generations: int = 100
    elite_size: int = 5         # Los mejores pasan directo
    mutation_rate: float = 0.2
    mutation_strength: float = 1.0
    # Configuración para la "Gravedad" interna
    gravity_steps_per_gen: int = 20  # Pocos pasos, solo para acomodar
    gravity_final_steps: int = 1000  # Muchos pasos al final para el mejor
