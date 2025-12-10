"""
Script to refine/tune an existing solution.
Loads a solution CSV and applies intensive optimization to compact it further.

Usage: python tune_solution.py <solution_id>
Example: python tune_solution.py 10
"""
import sys
from pathlib import Path

# Add root dir to path
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import SCALE_FACTOR
from app.visualization import TreeVisualizer
from app.algorithms import PackingAlgorithm
from app.utils.io import load_solution, save_solution

def tune_solution(solution_id, iterations=10000, target_ratio=None, visualize=True):
    """
    Refina una solución existente mediante optimización intensiva.
    
    Args:
        solution_id: ID de la solución (ej: 10 para T10.csv)
        iterations: Número de iteraciones de optimización
        target_ratio: Ratio objetivo para la jaula (None = sin jaula)
        visualize: Si mostrar visualización al final
    """
    solution_file = f"solutions/T{solution_id}.csv"
    
    print(f"\n{'='*60}")
    print(f"TUNING DE SOLUCIÓN: T{solution_id}")
    print(f"{'='*60}\n")
    
    # Cargar solución existente
    print(f"-> Cargando {solution_file}...")
    try:
        trees = load_solution(solution_file)
        print(f"-> Cargados {len(trees)} árboles")
    except FileNotFoundError:
        print(f"✗ Error: No se encontró {solution_file}")
        return
    
    # Inicializar solver
    solver = PackingAlgorithm(SCALE_FACTOR)
    solver.set_initial_state(trees)
    
    # Score inicial
    initial_score = solver.evaluator.calculate_kaggle_score(trees)
    print(f"-> Score inicial (Metric): {initial_score:.6f}")
    
    # Optimización intensiva
    print(f"\n-> Iniciando optimización intensiva ({iterations} iteraciones)...")
    print("   Todos los árboles son mutables para máxima flexibilidad")
    
    # Hacer todos los árboles mutables
    for tree in solver.trees:
        tree.fixed = False
    
    # Crear config personalizado para tuning
    from app.algorithms.config import AnnealingConfig
    
    tuning_config = AnnealingConfig()
    tuning_config.alpha = 0.98             # Enfriamiento más lento
    tuning_config.min_temp = 0.001         # Temperatura mínima más baja
    tuning_config.viz_frames = 0           # Sin visualización durante optimización
    
    # Recrear solver con config personalizado
    solver = PackingAlgorithm(SCALE_FACTOR, config=tuning_config)
    solver.set_initial_state(trees)
    
    # Configurar Jaula si se especifica
    if target_ratio:
        solver.set_cage_from_ratio(target_ratio)
    
    # Hacer todos los árboles mutables de nuevo
    for tree in solver.trees:
        tree.fixed = False
    
    # Habilitar visualización si se solicita
    if visualize:
        tuning_config.viz_frames = 50  # Mostrar 50 frames durante optimización
        viz = TreeVisualizer(SCALE_FACTOR)
    else:
        viz = None
    
    # Ejecutar annealing directamente con parámetros agresivos
    print(f"   Parámetros: temp=20.0, mag=3.0, allow_area_growth=False")
    if visualize:
        print(f"   Visualización: ACTIVADA (mostrando {tuning_config.viz_frames} frames)")
    
    success = solver._run_annealing(
        iterations=iterations,
        initial_temp=20.0,        # Temperatura alta para exploración
        mag_factor=3.0,           # Magnitud grande para movimientos amplios
        visualizer=viz if visualize else None,
        allow_area_growth=False   # NO permitir crecimiento, solo compactación
    )
    
    if success:
        print("   ✓ Optimización completada exitosamente")
    else:
        print("   ⚠ Optimización terminó sin encontrar mejora")
    
    final_trees = solver.trees
    
    # Score final
    final_score = solver.evaluator.calculate_kaggle_score(final_trees)
    improvement = ((initial_score - final_score) / initial_score) * 100
    
    print(f"\n{'='*60}")
    print("RESULTADOS DEL TUNING")
    print(f"{'='*60}")
    print(f"Score inicial: {initial_score:.6f}")
    print(f"Score final:   {final_score:.6f}")
    
    if final_score < initial_score:
        print(f"✓ MEJORA: {improvement:.2f}% mejor")
        
        # Guardar solución mejorada
        backup_file = f"solutions/T{solution_id}_backup.csv"
        print(f"\n-> Guardando backup en {backup_file}")
        save_solution(backup_file, trees, initial_score)
        
        print(f"-> Sobrescribiendo {solution_file} con solución mejorada")
        save_solution(solution_file, final_trees, final_score)
        
    elif final_score > initial_score:
        print(f"✗ PEOR: {-improvement:.2f}% peor - No se guardará")
    else:
        print("= SIN CAMBIO: Score idéntico")
    
    print(f"{'='*60}\n")
    
    # Visualizar resultado
    if visualize and final_trees:
        print("-> Mostrando visualización...")
        viz = TreeVisualizer(SCALE_FACTOR)
        viz.plot(final_trees, check_collisions=solver.detect_collisions)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python tune_solution.py <solution_id> [iterations] [target_ratio]")
        print("Ejemplo: python tune_solution.py 10")
        print("         python tune_solution.py 10 20000")
        print("         python tune_solution.py 10 20000 1.5")
        sys.exit(1)
    
    solution_id = int(sys.argv[1])
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    target_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    tune_solution(solution_id, iterations=iterations, target_ratio=target_ratio)
