import os
import glob
import re
import matplotlib.pyplot as plt
from app.utils.io import load_solution
from app.evaluation.cost_evaluator import CostEvaluator
from app.config import SCALE_FACTOR

def plot_scores():
    solutions_dir = "solutions"
    pattern = os.path.join(solutions_dir, "T*.csv")
    files = glob.glob(pattern)
    
    data = []
    
    evaluator = CostEvaluator(SCALE_FACTOR)
    
    print(f"Buscando soluciones en {solutions_dir}...")
    
    for file_path in files:
        # Ignorar backups
        if "_backup" in file_path:
            continue
            
        # Extraer N del nombre
        filename = os.path.basename(file_path)
        match = re.search(r"T(\d+)\.csv", filename)
        if match:
            n = int(match.group(1))
            
            try:
                trees = load_solution(file_path)
                # Calcular score real
                score = evaluator.calculate_kaggle_score(trees)
                data.append((n, score))
                print(f"-> T{n}: Score {score:.4f}")
            except Exception as e:
                print(f"Error leyendo {filename}: {e}")
    
    if not data:
        print("No se encontraron soluciones válidas.")
        return

    # Ordenar por N
    data.sort(key=lambda x: x[0])
    
    ns = [d[0] for d in data]
    scores = [d[1] for d in data]
    
    total_score = sum(scores)
    
    plt.figure(figsize=(12, 6))
    plt.plot(ns, scores, 'o-', markersize=4, alpha=0.7, label='Score (Area/N)')
    
    # Línea de tendencia suave podría ser útil, pero 'o-' basta
    
    plt.title(f"Evolución del Score (Area/N) | Score Final: {total_score:.6f}")
    plt.xlabel("N (Número de árboles)")
    plt.ylabel("Score (Area / N)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    output_file = "score_evolution.png"
    plt.savefig(output_file)
    print(f"\nGráfico guardado en {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_scores()
