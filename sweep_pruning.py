import os
import subprocess
import re

# Configuración
BIN = "./rust_optimizer/target/release/christmas_tree_optimizer"
SOL_DIR = "./solutions"

def run_broad_pruning():
    # 1. Asegurar que estamos en el sitio correcto y el binario existe
    if not os.path.exists(BIN):
        print("🔧 Binario no encontrado. Compilando Rust...")
        subprocess.run(["cargo", "build", "--release"], cwd="rust_optimizer")

    print("🚀 Iniciando barrido global de Prunning (1 a 200)...")

    # Listar todos los archivos disponibles una vez
    all_files = os.listdir(SOL_DIR)
    
    # Patrón para identificar archivos T{N}...
    pattern = re.compile(r"^T(\d+).*?\.csv$")

    # Recorremos cada N objetivo (de 199 hacia abajo)
    for n in range(199, 1, -1):
        target_file = f"{SOL_DIR}/T{n}-1.csv"
        print(f"\n🎯 Buscando la mejor poda para T{n}...")
        
        found_any = False

        # Para este N, probamos TODAS las fuentes superiores disponibles
        # Filtramos archivos que tengan un N mayor al objetivo
        candidates = []
        for f in all_files:
            match = pattern.match(f)
            if match:
                src_n = int(match.group(1))
                if src_n > n:
                    candidates.append(f)
        
        # Sort candidates naturally? Maybe unnecessary but nice
        candidates.sort()

        for src_file_name in candidates:
            src_file = os.path.join(SOL_DIR, src_file_name)
            
            # Ejecutar poda
            cmd = [
                BIN,
                "-i", src_file,
                "-o", target_file,
                "--target-n", str(n),
                "--strategy", "pruning"
            ]
            
            # Ejecutamos. Rust ya se encarga de comparar y guardar si es mejor.
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if "✨ ¡NUEVO RÉCORD!" in result.stdout:
                found_any = True
                print(f"   ✨ Mejora encontrada usando {src_file_name} como fuente!")

        if not found_any and not os.path.exists(target_file):
            print(f"   ⚠️ No se encontraron fuentes útiles para T{n}")
        elif os.path.exists(target_file):
            # Opcional: Actualizar el listado de archivos si target_file es nuevo? 
            # No es estrictamente necesario para pruning descendente si solo usamos src_n > n
            pass

if __name__ == "__main__":
    run_broad_pruning()
