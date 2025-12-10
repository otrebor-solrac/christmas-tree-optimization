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

# --- Configuration Data ---
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

# --- Component: IO & State Management ---
class SolutionManager:
    """Encarga de la persistencia y comparación de soluciones."""
    
    def __init__(self, solutions_dir="solutions"):
        self.solutions_dir = Path(solutions_dir)
        self.solutions_dir.mkdir(exist_ok=True)

    def get_path(self, n):
        return self.solutions_dir / f"T{n}.csv"

    def try_save_improvement(self, solver, candidate_trees, n, strategy_name="Optimización"):
        """Retorna True si la solución es mejor y se guardó."""
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

# --- Component: Algorithms ---
class Compactor:
    """Módulo de post-procesamiento físico."""
    
    @staticmethod
    def apply_gradual_compaction(solver, trees, steps=100):
        if len(trees) < 2: return trees
        
        # Helper para centro de masa
        def calc_com(t_list):
            if not t_list: return 0.0, 0.0
            cx = sum(float(t.center_x) for t in t_list) / len(t_list)
            cy = sum(float(t.center_y) for t in t_list) / len(t_list)
            return cx, cy

        initial_score = solver.evaluator.calculate_kaggle_score(trees)
        best_trees = copy.deepcopy(trees)
        best_score = initial_score
        improvements = 0
        solver.trees = trees # Bind state

        for _ in range(steps):
            com_x, com_y = calc_com(solver.trees)
            
            # Ordenar: lejanos primero
            dists = sorted(
                [(math.hypot(float(t.center_x)-com_x, float(t.center_y)-com_y), i) 
                 for i, t in enumerate(solver.trees)], 
                key=lambda x: x[0], reverse=True
            )
            
            for dist, i in dists:
                if dist < 0.001: continue
                t = solver.trees[i]
                
                # Física simple: mover hacia el centro con ruido
                dx, dy = com_x - float(t.center_x), com_y - float(t.center_y)
                angle = math.atan2(dy, dx) + random.uniform(-0.8, 0.8)
                step = 0.01
                
                old_state = (t.center_x, t.center_y)
                t.center_x += Decimal(math.cos(angle) * step)
                t.center_y += Decimal(math.sin(angle) * step)
                t.update_polygon()
                
                others = solver.trees[:i] + solver.trees[i+1:]
                if solver._check_strict_collision(t, others):
                    t.center_x, t.center_y = old_state
                    t.update_polygon()
            
            new_score = solver.evaluator.calculate_kaggle_score(solver.trees)
            if new_score < best_score:
                best_score = new_score
                best_trees = copy.deepcopy(solver.trees)
                improvements += 1
        
        if improvements > 0:
            print(f"   ✓ Compactación: {improvements} mejoras. {initial_score:.6f} -> {best_score:.6f}")
            return best_trees
        return trees

class StrategyExecutor:
    """Orquestador de estrategias de optimización."""
    
    def __init__(self, solver, visualizer, manager: SolutionManager, config: AppConfig):
        self.solver = solver
        self.viz = visualizer
        self.manager = manager
        self.config = config

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
            if self.solver.prune_solution_from_n_plus_1(target_n, visualizer=self.viz, optimize=optimize):
                print(f"   [Pruning] Éxito derivando T{target_n} desde T{next_n}")
                self.manager.try_save_improvement(self.solver, self.solver.trees, target_n, "Pruning")
                return True
        return False

    def _attempt_geometric(self, target_n):
        # [NEW] Multiplo de 4 Strategy
        if target_n % 4 == 0:
            # 2x2 Mosaic (4 tiles of N/4)
            n_base = target_n // 4
            if self.solver.generate_custom_mosaic(target_n, n_base, 2, 2, visualizer=self.viz):
                self.manager.try_save_improvement(self.solver, self.solver.trees, target_n, "Mosaic4x")
            return True  # Exclusive: stop here regardless of outcome

        # A. Mosaico Cuadrado
        # k_sq = math.sqrt(target_n)
        # if k_sq.is_integer() and k_sq > 1:
        #     if self.solver.generate_square_mosaic_solution(target_n, int(k_sq), visualizer=self.viz):
        #         saved |= self.manager.try_save_improvement(self.solver, self.solver.trees, target_n, "MosaicSq")

        # B. Grid Zipper
        # Buscar factores (w, h) tal que w*h = N y w/h ≈ 2 (rango 1.7 a 2.4)
        zipper_dims = None
        zipper_candidates = []
        for h in range(1, int(math.sqrt(target_n)) + 1):
            if target_n % h == 0:
                w = target_n // h
                ratio = w / h
                if 1.7 <= ratio <= 2.4:
                    zipper_candidates.append((w, h, ratio))

        if zipper_candidates:
            # Priorizar ratios <= 2.0. Sort key: (is_above_2, distance_to_2)
            zipper_candidates.sort(key=lambda x: (1 if x[2] > 2.0 else 0, abs(x[2] - 2.0)))
            zipper_dims = (zipper_candidates[0][0], zipper_candidates[0][1])

        if zipper_dims:
            # Pasamos rows=h, cols=w. Asegúrate de actualizar generate_grid_zipper_solution para aceptar estos argumentos.
            if self.solver.generate_grid_zipper_solution(target_n, visualizer=self.viz, rows=zipper_dims[1], cols=zipper_dims[0]):
                if self.manager.try_save_improvement(self.solver, self.solver.trees, target_n, "GridZipper"):
                    return True

        # C. Tessellation (Patrones hexagonales)
        k_check_2 = math.sqrt(target_n / 2)
        k_check_3 = math.sqrt(target_n / 3)
        use_tess = (target_n == 4 or (target_n > 3 and (k_check_2.is_integer() or k_check_3.is_integer())))
        
        if use_tess:
             if self.solver.generate_tessellated_solution(target_n, visualizer=self.viz):
                 if self.manager.try_save_improvement(self.solver, self.solver.trees, target_n, "Tessellation"):
                     return True
        
        return False

    def _attempt_incremental(self, target_n):
        prev_n = target_n - 1
        prev_file = self.manager.get_path(prev_n)
        
        should_restart = False
        
        # Cargar estado previo
        if prev_n > 0:
            try:
                # print(f"-> Buscando solución anterior: {prev_file}")
                trees = load_solution(prev_file)
                self.solver.set_initial_state(trees)
            except FileNotFoundError:
                print(f"ERROR: No se encontró {prev_file} para construir T{target_n}.")
                return None

        # Verificar si ya tenemos los árboles (caso de re-optimización)
        n_new = target_n - len(self.solver.trees)
        if n_new <= 0 and n_new != 0: 
            return None # Caso borde

        # Resolver
        strat = "random" if should_restart else "radial"
        final_trees = self.solver.add_trees(n_new=n_new, visualizer=self.viz, strategy=strat)
        
        # Refinar
        final_trees = Compactor.apply_gradual_compaction(self.solver, final_trees)
        
        # Guardar
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

def run_gradual_mode(config: AppConfig, manager: SolutionManager, viz):
    """Wrapper para importar y ejecutar el modo de tuning intensivo."""
    try:
        from tune_gradual import gradual_compaction
    except ImportError:
        print("Error: No se encuentra 'tune_gradual.py'.")
        return

    print(f"Iniciando Tuning Gradual para N={config.target_n}...")
    path = manager.get_path(config.target_n)
    trees = load_solution(path)
    
    if not trees:
        print(f"Error: No existe {path} para optimizar.")
        return

    solver = PackingAlgorithm(SCALE_FACTOR)
    solver.trees = trees
    solver._refresh_current_score()
    gradual_compaction(config.target_n, visualize=viz, steps=5000)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Christmas Tree Packing Solver")
    
    # Argumentos principales
    parser.add_argument("--target", type=int, default=200, help="Target N trees (default: 200)")
    parser.add_argument("--mode", type=str, default="NORMAL", choices=["NORMAL", "GRADUAL"], help="Execution mode")
    
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
    
    # Resolver defaults y detectar rango explícito
    default_start = 87
    default_end = 200
    explicit_range = (args.start is not None) or (args.end is not None)
    
    b_start = args.start if args.start is not None else default_start
    b_end = args.end if args.end is not None else default_end

    # Crear configuración
    config = AppConfig(
        target_n=args.target,
        batch_mode=args.batch,
        batch_start=b_start,
        batch_end=b_end,
        mode=args.mode,
        visualize=args.viz,
        use_pruning=args.pruning,
        skip_tuning=args.no_tuning
    )

    # Inicializar componentes
    manager = SolutionManager()
    viz = TreeVisualizer(SCALE_FACTOR) if config.visualize else None
    
    # Modo Especial: Gradual Tuning
    if config.mode == "GRADUAL":
        run_gradual_mode(config, manager, viz)
        return

    # Modo Estándar / Batch
    solver = PackingAlgorithm(SCALE_FACTOR)
    executor = StrategyExecutor(solver, viz, manager, config)

    if config.use_pruning:
        # Lógica especial de pruning inverso
        # Detectar si el usuario invirtió start/end (ej: 113 -> 99) para indicar rango sin flag --batch
        if config.batch_start > config.batch_end:
            start_n = config.batch_start
            end_n = config.batch_end
        elif config.batch_mode or explicit_range:
            start_n = max(config.batch_start, config.batch_end)
            end_n = min(config.batch_start, config.batch_end)
        else:
            start_n = config.target_n
            end_n = 3
            
        print(f"PRUNING CASCADE: {start_n} -> {end_n}")
        
        for n in range(start_n, end_n - 1, -1):
            print(f"\n--- Pruning T{n} desde T{n+1} ---")
            executor.execute_step(n)

    elif config.batch_mode:
        run_batch(executor, config.batch_start, config.batch_end)
        
    else:
        trees = executor.execute_step(config.target_n)
        if trees and config.visualize:
            print("Visualizando resultado final...")
            viz.plot(trees, score_real=solver.best_score, check_collisions=solver.detect_collisions)

if __name__ == "__main__":
    # python main.py --target 150 --viz
    # python main.py --batch --start 50 --end 100 --viz
    # python main.py --pruning --target 200
    # python main.py --pruning
    #  python main.py --pruning --batch --start 100 --end 150
    # python main.py --pruning --target 200 --no-tuning 
    main()