#!/usr/bin/env python3
"""
Script para generar el archivo de submission para Kaggle
Combina todos los archivos T{N}.csv en un solo archivo con el formato requerido.
"""

import os
import csv
from pathlib import Path

def create_submission(solutions_dir: str = "solutions_compress", output_file: str = "submission2.csv"):
    """
    Crea el archivo de submission combinando todos los T{N}.csv
    
    Formato de salida:
    id,x,y,deg
    001_0,s0.0,s0.0,s90.0
    001_1,s0.202736,s-0.511271,s90.0
    ...
    """
    
    solutions_path = Path(solutions_dir)
    output_path = Path(output_file)
    
    # Recolectar todas las soluciones
    all_rows = []
    missing = []
    
    for n in range(1, 201):  # T1 a T200
        file_path = solutions_path / f"T{n}.csv"
        
        if not file_path.exists():
            missing.append(n)
            print(f"⚠️  Falta: T{n}.csv")
            continue
        
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                trees = list(reader)
                
                if len(trees) != n:
                    print(f"⚠️  T{n}.csv tiene {len(trees)} árboles (esperados: {n})")
                
                for i, tree in enumerate(trees):
                    # Formato de id: NNN_I (puzzle padded a 3 dígitos, índice del árbol)
                    puzzle_id = f"{n:03d}_{i}"
                    
                    # Las coordenadas ya vienen con prefijo 's' en los archivos
                    x = tree['x']
                    y = tree['y']
                    deg = tree['deg']
                    
                    all_rows.append({
                        'id': puzzle_id,
                        'x': x,
                        'y': y,
                        'deg': deg
                    })
                    
        except Exception as e:
            print(f"❌ Error leyendo T{n}.csv: {e}")
            missing.append(n)
    
    # Ordenar por id
    all_rows.sort(key=lambda r: r['id'])
    
    # Escribir archivo de submission
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'x', 'y', 'deg'])
        writer.writeheader()
        writer.writerows(all_rows)
    
    # Resumen
    total_trees = sum(range(1, 201))  # 1+2+3+...+200 = 20100
    print(f"\n✅ Submission creado: {output_path}")
    print(f"   Total de filas: {len(all_rows)} / {total_trees} esperadas")
    
    if missing:
        print(f"   ⚠️  Archivos faltantes: {len(missing)}")
        print(f"      {missing[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        print(f"   ✅ Todos los archivos presentes (T1 a T200)")
    
    return len(all_rows), len(missing)


def validate_submission(submission_file: str = "submission.csv"):
    """Valida el formato del archivo de submission"""
    
    with open(submission_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\n📋 Validando {submission_file}...")
    print(f"   Filas: {len(rows)}")
    print(f"   Columnas: {list(rows[0].keys()) if rows else 'N/A'}")
    
    # Verificar formato de id
    sample = rows[:5] if rows else []
    print(f"   Muestra:")
    for row in sample:
        print(f"      {row['id']}, {row['x'][:15]}..., {row['y'][:15]}..., {row['deg'][:15]}...")
    
    # Verificar que todos los x, y, deg empiezan con 's'
    invalid = [r for r in rows if not (r['x'].startswith('s') and r['y'].startswith('s') and r['deg'].startswith('s'))]
    if invalid:
        print(f"   ⚠️  {len(invalid)} filas sin prefijo 's'")
    else:
        print(f"   ✅ Formato válido (prefijo 's' correcto)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Genera submission para Kaggle")
    parser.add_argument("--solutions", default="solutions", help="Directorio con los archivos T{N}.csv")
    parser.add_argument("--output", default="submission.csv", help="Archivo de salida")
    parser.add_argument("--validate", action="store_true", help="Solo validar un archivo existente")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_submission(args.output)
    else:
        create_submission(args.solutions, args.output)
        validate_submission(args.output)
