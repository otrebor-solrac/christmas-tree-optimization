import os
import glob
import re
import math
import pandas as pd

def extract_n_from_filename(filename):
    # Match EXACTLY T{Integer}.csv
    match = re.search(r'^T(\d+)\.csv$', filename)
    if match:
        return int(match.group(1))
    return None

def get_data_from_csv(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None
            # Leer score de la primera línea de datos (line 1, col 4)
            # id,x,y,deg,score
            parts = lines[1].strip().split(',')
            if len(parts) >= 5:
                score_str = parts[4].lstrip('s')
                return float(score_str)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

def generate_report(solutions_dir="solutions", output_file="report_optimization.csv"):
    files = glob.glob(os.path.join(solutions_dir, "T*.csv"))
    
    data = []
    
    print(f"🔍 Encontrados {len(files)} archivos en {solutions_dir}...")
    
    for fw in files:
        n = extract_n_from_filename(os.path.basename(fw))
        if n is None:
            continue
            
        score = get_data_from_csv(fw)
        if score is None:
            continue
            
        # Calcular Lado (Side) basado en Score = (Side^2) / N
        # => Side = sqrt(Score * N)
        side = math.sqrt(score * n)
        
        data.append({
            "N": n,
            "Side": side,
            "Score": score
        })
    
    # Ordenar por N
    data.sort(key=lambda x: x["N"])
    
    # Calcular Diferencias
    report_data = []
    prev_score = None
    
    for item in data:
        n = item["N"]
        score = item["Score"]
        side = item["Side"]
        
        diff = 0.0
        if prev_score is not None:
            diff = score - prev_score
            
        report_data.append({
            "N arboles": n,
            "Lado": side,
            "Score Kaggle": score,
            "Diferencia": diff
        })
        
        prev_score = score
        
    df = pd.DataFrame(report_data)
    
    # Guardar CSV
    output_path = os.path.join(solutions_dir, output_file)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Reporte generado: {output_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    generate_report()
