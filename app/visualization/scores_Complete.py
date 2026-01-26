import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys

# Configurar path para permitir imports absolutos de 'app'
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.utils.io import load_solution
from app.evaluation.cost_evaluator import CostEvaluator
from app.config import SCALE_FACTOR

def collect_scores(directory, mode):
    """
    mode: 'std' for TN.csv, 'dash1' for TN-1.csv
    """
    path = Path(directory)
    if not path.exists():
        print(f"Directory {directory} not found.")
        return []

    evaluator = CostEvaluator(SCALE_FACTOR)
    results = []
    
    # Regex patterns
    re_std = re.compile(r"^T(\d+)\.csv$")
    re_dash1 = re.compile(r"^T(\d+)-1\.csv$")

    print(f"Buscando en {directory} (modo: {mode})...")

    for file in path.glob("T*.csv"):
        if "_backup" in file.name: continue
        
        n = None
        if mode == 'std':
            match = re_std.match(file.name)
            if match: n = int(match.group(1))
        elif mode == 'dash1':
            match = re_dash1.match(file.name)
            if match: n = int(match.group(1))

        if n is not None:
            try:
                trees = load_solution(file)
                score = evaluator.calculate_kaggle_score(trees)
                results.append((n, score))
            except Exception as e:
                print(f"Error leyendo {file.name}: {e}")
    
    results.sort(key=lambda x: x[0])
    return results

def plot_complete_comparison():
    # 1. Recolectar datos
    print("--- Recolectando Solutions (TN.csv strictly) ---")
    data_solutions_std = collect_scores("solutions", "std")

    print("--- Recolectando Solutions (T*-1.csv) ---")
    data_solutions_1 = collect_scores("solutions", "dash1")

    plt.figure(figsize=(15, 8))

    # Graficar 'solutions' estándar
    if data_solutions_std:
        ns, scores = zip(*data_solutions_std)
        plt.plot(ns, scores, 'x-', color='red', markersize=6, alpha=0.7, label='Base Solutions (TN)')
        print(f"Solutions (TN): {len(data_solutions_std)} puntos.")

    # Graficar 'solutions' (T*-1) - Los nuevos de pruning
    if data_solutions_1:
        ns, scores = zip(*data_solutions_1)
        plt.plot(ns, scores, 'o--', color='orange', markersize=4, alpha=0.8, label='Pruning (TN-1)')
        print(f"Solutions (TN-1): {len(data_solutions_1)} puntos.")

    plt.title("Comparativa de Scores: Base (TN) vs Pruning (TN-1)")
    plt.xlabel("Number of Trees (N)")
    plt.ylabel("Score (Area/N)")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.yscale('log')
    plt.legend()
    
    # --- COMPARATIVA DE MEJORAS ---
    print("\n" + "="*50)
    print(f"{'N':<5} | {'Base (TN)':<12} | {'Pruning (TN-1)':<15} | {'Mejora':<10}")
    print("-" * 50)
    
    # Convertir a diccionarios para búsqueda rápida
    dict_std = dict(data_solutions_std)
    dict_1 = dict(data_solutions_1)
    
    improvements = 0
    # Recorrer todos los N que están en ambos
    all_ns = sorted(set(dict_std.keys()) | set(dict_1.keys()))
    for n in all_ns:
        s_std = dict_std.get(n)
        s_1 = dict_1.get(n)
        
        if s_std is not None and s_1 is not None:
            # En Kaggle, menor es mejor.
            if s_1 < s_std:
                diff = s_std - s_1
                print(f"{n:<5} | {s_std:<12.6f} | {s_1:<15.6f} | ✅ {diff:.6f}")
                improvements += 1
    
    if improvements == 0:
        print("No se encontraron casos donde T(N)-1 sea mejor que T(N).")
    else:
        print(f"\nSe encontraron {improvements} mejoras en archivos T(N)-1.")
    print("="*50 + "\n")
    output_file = "score_comparison_pruning.png"
    plt.savefig(output_file, dpi=200)
    print(f"\nGráfico guardado en {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_complete_comparison()
