#!/usr/bin/env python
import subprocess
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Compila y ejecuta el optimizador Rust")
    parser.add_argument("--target", type=int, required=True, help="Número de árboles (N) a optimizar (ej: 20)")
    parser.add_argument("--pop", type=int, default=100, help="Tamaño de la población (default: 100)")
    parser.add_argument("--gen", type=int, default=10, help="Número de generaciones (default: 200)")
    parser.add_argument("--steps", type=int, default=50, help="Pasos de gravedad por generación (default: 50)")
    parser.add_argument("--groups", action="store_true", help="Activar modo de grupos (pares)")
    parser.add_argument("--fine-tune", type=int, default=0, help="Iteraciones de Fine-Tuning al final (default: 0)")
    parser.add_argument("--strategy", type=str, default="all", help="Estrategia: all, mosaic, zipper, pruning, incremental, ga")
    parser.add_argument("--optimizer", type=str, default="none", help="Optimizador: none (skip), ga, cmaes, o sa")
    
    args = parser.parse_args()
    
    # Rutas
    root_dir = Path(__file__).parent.resolve()
    rust_dir = root_dir / "rust_optimizer"
    solutions_dir = root_dir / "solutions"
    binary_path = rust_dir / "target" / "release" / "christmas_tree_optimizer"
    
    # Archivo de solución (output siempre será T{N}.csv)
    output_file = solutions_dir / f"T{args.target}.csv"
    
    # 1. Compilar Rust (Release mode)
    print(f"🔨 Compilando proyecto Rust...")
    try:
        subprocess.run(["cargo", "build", "--release"], cwd=rust_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("❌ Error en la compilación de Rust.")
        sys.exit(1)

    # 2. Ejecutar Binario
    print(f"\n🚀 Iniciando Pipeline Rust para T{args.target}...")
    
    cmd = [
        str(binary_path),
        "--input", str(output_file), # Usamos el mismo como entrada (Rust manejará si no existe)
        "--output", str(output_file),
        "--target-n", str(args.target),
        "--pop-size", str(args.pop),
        "--generations", str(args.gen),
        "--gravity-steps", str(args.steps),
        "--strategy", args.strategy,
        "--optimizer", args.optimizer
    ]
    
    if args.groups:
        cmd.append("--use-groups")
        
    if args.fine_tune > 0:
        cmd.extend(["--fine-tune-iters", str(args.fine_tune)])
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
