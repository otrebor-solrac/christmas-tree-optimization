import matplotlib.pyplot as plt
from pathlib import Path
import re
import csv
import sys
import math

# Configurar path para permitir imports absolutos de 'app'
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def plot_scores(solutions_dir="solutions"):
    path = Path(solutions_dir)
    if not path.exists():
        print(f"Directory {solutions_dir} not found.")
        return

    results = []
    
    print(f"🚀 Buscando soluciones en {solutions_dir} (Modo Ultra Rápido sin Polars)...")
    
    for file in path.glob("T*.csv"):
        if "_backup" in file.name: continue
        
        match = re.match(r"T(\d+)(?:-.*)?\.csv", file.name) # Updated regex to match T44-2.csv etc if needed, but original was just T(\d+).csv. Wait.
        # Original regex: r"T(\d+)\.csv"
        # File list showed: T100-2.csv, T100.csv
        # The original code likely skipped T100-2.csv.
        # Let's keep original regex behavior or improve it?
        # User pattern seems to be T{N}.csv for "official" ones?
        # The file listing showed T100-2.csv.
        # Only T(\d+).csv matches T1.csv, T100.csv. It does NOT match T100-2.csv.
        # If I want to verify, I should check if the user wants *all* variants or just the main ones.
        # Given the graph typically plots one point per N, skipping variants might be intended or accidental.
        # However, looking at the previous file content:
        # 29:         match = re.match(r"T(\d+)\.csv", file.name)
        # It strictly matched T[digits].csv.
        # I will stick to the original behavior to avoid plotting duplicates for the same N.
        
        match = re.match(r"T(\d+)\.csv", file.name)
        if match:
            n = int(match.group(1))
            try:
                # OPTIMIZACIÓN: Leer solo la primera línea de datos
                with open(file, 'r') as f:
                    reader = csv.DictReader(f)
                    first_row = next(reader, None)
                    if first_row and 'score' in first_row:
                        # El score tiene prefijo 's'
                        score_str = first_row['score'].lstrip('s')
                        score = float(score_str)
                        results.append((n, score))
                        # print(f"-> T{n}: Score {score:.4f}") # Reducir ruido en consola
            except Exception as e:
                print(f"Error leyendo {file.name}: {e}")
    
    if not results:
        print("No solutions found or could not parse scores.")
        return

    results.sort(key=lambda x: x[0])
    ns, scores = zip(*results)
    
    # Calcular lado (Max Side = sqrt(Score * N))
    sides = [math.sqrt(s * n) for s, n in zip(scores, ns)]
    
    total_score = sum(scores)
    
    print(f"✅ Procesados {len(results)} archivos instantáneamente.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Score (Area/N)
    ax1.plot(ns, scores, 'o-', color='#2ecc71', markersize=4, alpha=0.7, label='Score (Area/N)')
    ax1.set_title(f"Evolución del Score (Area/N) | Score Final Total: {total_score:.6f}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Score (Area / N)", fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Plot 2: Side Length (Max Side)
    ax2.plot(ns, sides, 's-', color='#3498db', markersize=4, alpha=0.7, label='Lado del Cuadrado (Max Side)')
    ax2.set_title(r"Evolución del Lado del Cuadrado (L = $\sqrt{S \cdot N}$)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Number of Trees (N)", fontsize=12)
    ax2.set_ylabel("Largo del Lado", fontsize=12)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    
    output_file = "score_evolution.png"
    plt.savefig(output_file, dpi=150)
    print(f"Gráfico guardado en {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_scores()
