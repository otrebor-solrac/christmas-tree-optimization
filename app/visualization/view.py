"""
Script to view a specific solution CSV.
Usage: python -m app.visualization.view <solution_id>
"""
import argparse
import sys
from pathlib import Path

# Ajuste de path para ejecución directa
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    __package__ = "app.visualization"

from ..config import SCALE_FACTOR
from .visualizer import TreeVisualizer
from ..algorithms.packing import PackingAlgorithm
from ..utils.io import load_solution

def view_solution_file(csv_path):
    print(f"-> Cargando solución: {csv_path}")
    
    try:
        trees = load_solution(csv_path)
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return

    if not trees:
        print("El archivo no contiene árboles.")
        return

    # Usamos PackingAlgorithm para detección de colisiones
    solver = PackingAlgorithm(SCALE_FACTOR)
    viz = TreeVisualizer(SCALE_FACTOR)

    # Visualizar
    viz.plot(trees, check_collisions=solver.detect_collisions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizar una solución de empaquetado de árboles.')
    parser.add_argument('solution_id', type=int, help='ID de la solución (ej: 3 para ver solutions/T3.csv)')
    
    args = parser.parse_args()
    
    # Construir ruta automáticamente
    csv_path = f"solutions/T{args.solution_id}.csv"
    
    view_solution_file(csv_path)
