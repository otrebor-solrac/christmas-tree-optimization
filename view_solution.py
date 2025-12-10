"""
Script to view a specific solution CSV.
Usage: python view_solution.py <path_to_csv>
"""
import sys
import argparse
from pathlib import Path

# Add root dir to path
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import SCALE_FACTOR
from app.visualization import TreeVisualizer
from app.algorithms import PackingAlgorithm
from app.utils.io import load_solution

def view_solution(csv_path):
    print(f"-> Cargando solución: {csv_path}")
    
    # 1. Cargar árboles
    try:
        trees = load_solution(csv_path)
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return

    if not trees:
        print("El archivo no contiene árboles.")
        return

    # 2. Inicializar componentes para cálculo
    # Usamos PackingAlgorithm solo para acceder a sus métodos de detección de colisiones y métricas
    # (aunque podríamos usar CostEvaluator directamente, PackingAlgorithm ya tiene helpers)
    solver = PackingAlgorithm(SCALE_FACTOR)
    viz = TreeVisualizer(SCALE_FACTOR)

    # 3. Calcular métricas
    # Necesitamos pasar los árboles al solver para que detect_collisions funcione (o pasarle la lista)
    has_collisions, collision_pairs = solver.detect_collisions(trees)
    
    # 4. Visualizar
    # El método plot de TreeVisualizer ya calcula y muestra el área y la métrica Kaggle.
    # Le pasamos la función de detección de colisiones del solver.
    viz.plot(trees, check_collisions=solver.detect_collisions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizar una solución de empaquetado de árboles.')
    parser.add_argument('solution_id', type=int, help='ID de la solución (ej: 3 para ver solutions/T3.csv)')
    
    args = parser.parse_args()
    
    # Construir ruta automáticamente
    csv_path = f"solutions/T{args.solution_id}.csv"
    
    view_solution(csv_path)
