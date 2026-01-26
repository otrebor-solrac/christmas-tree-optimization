#!/usr/bin/env python3
import subprocess
import os
import sys

# --- CONFIGURACIÓN ---
START_N = 21
END_N = 21
STRATEGY = "woa"
GENERATIONS = 5000
POP_SIZE = 200
RUST_DIR = "rust_optimizer"

def run_optimization(n):
    print(f"\n🚀 [Batch] Optimizando T{n} con {STRATEGY}...")
    
    # Es mejor ser explícito con las rutas para que cargue la solución actual
    input_file = f"../solutions/T{n}.csv"
    output_file = f"../solutions/T{n}-3.csv"
    
    cmd = [
        "cargo", "run", "--bin", "christmas_tree_optimizer", "--release", "--",
        "--strategy", STRATEGY,
        "--target-n", str(n),
        "--generations", str(GENERATIONS),
        "--pop-size", str(POP_SIZE),
        "--pure-random",
        "-o", output_file
    ]
    
    try:
        # Ejecutamos desde el directorio de rust_optimizer
        subprocess.run(cmd, cwd=RUST_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en T{n}: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Proceso detenido por el usuario.")
        sys.exit(0)

def main():
    print(f"🌊 Iniciando Batch Whale Optimization (N={START_N} a {END_N})")
    
    # Asegurarse de que el binario esté compilado una sola vez al inicio
    print("🔨 Compilando optimizador...")
    subprocess.run(["cargo", "build", "--release"], cwd=RUST_DIR, check=True)
    
    for n in range(START_N, END_N + 1):
        run_optimization(n)
        
    print("\n✅ Batch completo.")

if __name__ == "__main__":
    main()
