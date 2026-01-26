"""
Main entry point for the Christmas Tree Packing application.
Refactored for modularity and maintainability.
"""
import sys
import math
import random
import time
import argparse
from pathlib import Path
from decimal import Decimal
from dataclasses import dataclass
import copy

# --- Setup Paths ---
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# --- Internal Imports ---
from app.config import SCALE_FACTOR
from app.visualization import TreeVisualizer
from app.algorithms.packing import PackingAlgorithm
from app.utils.io import save_solution, load_solution
from app.algorithms.gravity import GravityCompactor
from app.algorithms.strategies import SolutionGenerators
from app.algorithms.config import GravityConfig
from app.algorithms.config import GeneticConfig
from app.models.tree import ChristmasTree

@dataclass
class AppConfig:
    target_n: int
    batch_mode: bool
    batch_start: int
    batch_end: int
    mode: str  # "NORMAL", "GRADUAL"
    visualize: bool
    use_pruning: bool
    skip_tuning: bool

class SolutionManager:
    """Encarga de la persistencia y comparación de soluciones."""
    
    def __init__(self, solutions_dir="solutions"):
        self.solutions_dir = Path(solutions_dir)
        self.solutions_dir.mkdir(exist_ok=True)

    def get_path(self, n):
        return self.solutions_dir / f"T{n}.csv"

    def try_save_improvement(self, solver, candidate_trees, n, strategy_name="Optimización"):
        """
        Retorna True si la solución es mejor y se guardó.
        """
        filepath = self.get_path(n)
        current_score = solver.evaluator.calculate_kaggle_score(candidate_trees)
        existing_score = float('inf')

        if filepath.exists():
            try:
                ext_trees = load_solution(filepath)
                if ext_trees and len(ext_trees) == len(candidate_trees):
                    existing_score = solver.evaluator.calculate_kaggle_score(ext_trees)
            except Exception:
                pass

        if current_score < existing_score:
            print(f"   [{strategy_name}] MEJORA T{n}: {current_score:.6f} vs {existing_score:.6f}")
            save_solution(filepath, candidate_trees, current_score)
            return True
        
        return False

class StrategyExecutor:
    """Orquestador de estrategias de optimización."""
    
    def __init__(self, solver, visualizer, manager: SolutionManager, config: AppConfig):
        self.solver = solver
        self.viz = visualizer
        self.manager = manager
        self.config = config
        self.generator = SolutionGenerators(solver)

    def execute_step(self, target_n):
        """Pipeline principal de decisión."""
        # 1. Estrategia: Pruning (Top-Down)
        if self.config.use_pruning:
            if self._attempt_pruning(target_n):
                return self.solver.trees
            return None

        # 2. Estrategias Geométricas (Mosaicos, Tessellation)
        if self._attempt_geometric(target_n):
            return self.solver.trees

        # 3. Estrategia: Incremental (Bottom-Up)
        return self._attempt_incremental(target_n)

    def _attempt_pruning(self, target_n):
        next_n = target_n + 1
        next_path = self.manager.get_path(next_n)
        
        if next_path.exists():
            optimize = not self.config.skip_tuning
            if self.generator.prune_solution_from_n_plus_1(target_n, visualizer=self.viz, optimize=optimize):
                print(f"   [Pruning] Éxito derivando T{target_n} desde T{next_n}")
                self.manager.try_save_improvement(self.solver, self.solver.trees, target_n, "Pruning")
                return True
        return False

    def _attempt_geometric(self, target_n):
        """Delega la ejecución de estrategias geométricas al generador."""
        found_any = False
        for strat_name, trees in self.generator.attempt_geometric_strategies(target_n, self.viz):
            if self.manager.try_save_improvement(self.solver, trees, target_n, strat_name):
                found_any = True
                # Si es Mosaic4x, es exclusivo (según lógica original)
                if strat_name == "Mosaic4x": 
                    return True
                # Para otras estrategias, retornamos True si encontramos mejora, 
                # pero el generador podría yieldear más opciones si quisiéramos probar todas.
                # La lógica original retornaba al primer éxito de Zipper o Tessellation.
                return True
        return found_any

    def _attempt_incremental(self, target_n):
        """Estrategia incremental: Cargar N-1, añadir árbol, compactar."""
        prev_n = target_n - 1
        prev_path = self.manager.get_path(prev_n)
        prev_trees = None
        
        if prev_n > 0:
            if prev_path.exists():
                prev_trees = load_solution(prev_path)
            else:
                print(f"ERROR: No se encontró {prev_path} para construir T{target_n}.")
                return None

        final_trees = self.generator.generate_incremental_solution(target_n, prev_trees, self.viz)
        
        if final_trees:
            self.manager.try_save_improvement(self.solver, final_trees, target_n, "Incremental")
            
        return final_trees

# --- Main Application Logic ---
def run_batch(executor: StrategyExecutor, start_n, end_n):
    """Ejecuta un bucle de generación."""
    print(f"\n{'='*60}\nGENERACIÓN BATCH: {start_n} a {end_n}\n{'='*60}")
    total_start = time.time()
    
    for n in range(start_n, end_n + 1):
        step_start = time.time()
        print(f"\n[{n}/{end_n}] Procesando N = {n}...")
        
        try:
            trees = executor.execute_step(n)
            if trees:
                score = executor.solver.evaluator.calculate_kaggle_score(trees)
                print(f"✓ Hecho en {time.time() - step_start:.1f}s | Score: {score:.4f}")
            else:
                print(f"✗ Falló N={n}")
        except Exception as e:
            print(f"✗ Error crítico en N={n}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nBATCH COMPLETADO en {(time.time() - total_start)/60:.1f} min.")

def run_pruning_cascade(executor: StrategyExecutor, start_n, end_n):
    """Ejecuta la estrategia de pruning en orden descendente."""
    # Asegurar orden descendente (mayor a menor)
    high = max(start_n, end_n)
    low = min(start_n, end_n)
    
    print(f"PRUNING CASCADE: {high} -> {low}")
    for n in range(high, low - 1, -1):
        print(f"\n--- Pruning T{n} desde T{n+1} ---")
        executor.execute_step(n)

def run_gradual_mode(config: AppConfig, manager: SolutionManager, viz):
    """Wrapper para importar y ejecutar el modo de tuning intensivo."""
    from app.algorithms.gravity import GravityCompactor
    from app.algorithms.config import GravityConfig
    print(f"Iniciando Tuning Gradual para N={config.target_n}...")
    path = manager.get_path(config.target_n)
    trees = load_solution(path)
    
    if not trees:
        print(f"Error: No existe {path} para optimizar.")
        return

    g_config = GravityConfig(
        steps=5000, 
        visualize=config.visualize,
        enable_teleport=True
    )
    compactor = GravityCompactor(SCALE_FACTOR, g_config)
    final_trees = compactor.compact(trees, visualizer=viz)
    
    # Guardar si mejoró
    manager.try_save_improvement(compactor.solver, final_trees, config.target_n, "GradualMode")

def run_custom_mode(config: AppConfig, manager: SolutionManager, viz):
    """Ejecuta un patrón personalizado (ej: Grid con Shift manual)."""
    print(">>> EJECUTANDO MODO CUSTOM: Patrón 20x10 con Shift Manual <<<")
    
    # Parámetros de ajuste manual
    STRIDE_X = 0.412        
    ZIPPER_OFFSET_Y = 0.50 
    ROW_HEIGHT = 0.80     
    ROW_SHIFT = -0.15  # Ajuste manual para filas impares
    
    COLS = 20
    ROWS = 10
    
    solver = PackingAlgorithm(SCALE_FACTOR)
    solver.trees = []
    
    current_id = 1
    for row in range(ROWS):
        for col in range(COLS):
            x = col * STRIDE_X
            y = row * ROW_HEIGHT
            angle = 0
            
            # Zipper (columnas impares rotadas y desplazadas)
            if col % 2 != 0:
                angle = 180
                y += ZIPPER_OFFSET_Y
            
            # Stagger (desplazamiento en filas impares)
            if row % 2 != 0:
                x += ROW_SHIFT

            t = ChristmasTree(current_id, x, y, angle, fixed=False)
            solver.trees.append(t)
            current_id += 1
            
    solver.refresh_current_score()
    
    if viz:
        viz.plot(solver.trees, score_real=solver.best_score, check_collisions=solver.detect_collisions)

def run_genetic_mode(config: AppConfig, manager: SolutionManager, viz):
    """Ejecuta el algoritmo genético para una solución existente."""
    from app.algorithms.genetic import GeneticOptimizer
    
    print(f"Iniciando Modo Genético para N={config.target_n}...")
    path = manager.get_path(config.target_n)
    trees = load_solution(path)
    
    if not trees:
        print(f"Error: No existe {path} para optimizar. Genera una solución base primero.")
        return

    # Configuración del Genético
    gen_config = GeneticConfig(
        population_size=30,
        generations=50,
        gravity_steps_per_gen=50,
        gravity_final_steps=2000
    )
    
    optimizer = GeneticOptimizer(SCALE_FACTOR, gen_config)
    final_trees = optimizer.optimize(trees, manager=manager, target_n=config.target_n)
    
    manager.try_save_improvement(optimizer.compactor.solver, final_trees, config.target_n, "GeneticMode")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Christmas Tree Packing Solver")
    
    # Argumentos principales
    parser.add_argument("--target", type=int, default=200, help="Target N trees (default: 200)")
    parser.add_argument("--mode", type=str, default="NORMAL", choices=["NORMAL", "GRADUAL", "CUSTOM", "GENETIC"], help="Execution mode")
    
    # Flags booleanos
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    parser.add_argument("--pruning", action="store_true", help="Enable Pruning Cascade strategy")
    parser.add_argument("--no-tuning", action="store_true", help="Skip optimization in pruning")
    
    # Batch processing
    parser.add_argument("--batch", action="store_true", help="Run in batch mode")
    parser.add_argument("--start", type=int, default=None, help="Batch start N")
    parser.add_argument("--end", type=int, default=None, help="Batch end N")

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Crear configuración
    config = AppConfig(
        target_n=args.target,
        batch_mode=args.batch,
        batch_start=args.start if args.start is not None else 0,
        batch_end=args.end if args.end is not None else 200,
        mode=args.mode,
        visualize=args.viz,
        use_pruning=args.pruning,
        skip_tuning=args.no_tuning
    )

    # Inicializar componentes
    manager = SolutionManager()
    viz = TreeVisualizer(SCALE_FACTOR) if config.visualize else None
    
    # 1. MODO TUNING (Gradual)
    if config.mode == "GRADUAL":
        run_gradual_mode(config, manager, viz)
        return

    # 2. MODO CUSTOM (Patrón Manual)
    if config.mode == "CUSTOM":
        run_custom_mode(config, manager, viz)
        return

    # 3. MODO GENETIC (Optimización Global)
    if config.mode == "GENETIC":
        run_genetic_mode(config, manager, viz)
        return

    # Inicializar Solver para modos de generación
    solver = PackingAlgorithm(SCALE_FACTOR)
    executor = StrategyExecutor(solver, viz, manager, config)

    # 2. MODO PRUNING (Cascada descendente)
    if config.use_pruning:
        # Si se especificó rango o batch, usar esos límites. Si no, usar target -> 3
        has_range = (args.start is not None) or (args.end is not None) or config.batch_mode
        
        if has_range:
            run_pruning_cascade(executor, config.batch_start, config.batch_end)
        else:
            run_pruning_cascade(executor, config.target_n, 3)
        return

    # 3. MODO BATCH (Rango ascendente)
    if config.batch_mode:
        run_batch(executor, config.batch_start, config.batch_end)
        return
        
    # 4. MODO ESTÁNDAR (Single Target)
    trees = executor.execute_step(config.target_n)
    if trees and config.visualize:
        print("Visualizando resultado final...")
        viz.plot(trees, score_real=solver.best_score, check_collisions=solver.detect_collisions)

if __name__ == "__main__":
    # python main.py --target 150 --viz
    # 
    # python main.py --pruning --target 200
    # python main.py --pruning
    # python main.py --pruning --batch --start 100 --end 150
    # python main.py --pruning --target 200 --no-tuning 
    main()