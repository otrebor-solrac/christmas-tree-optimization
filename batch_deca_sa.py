import subprocess
import os

# Ajustes
BIN_PATH = "./target/release/christmas_tree_optimizer"
SOL_DIR = "../solutions"
START_N = 11
END_N = 40

def run_command(cmd):
    print(f"\n🚀 Ejecutando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    if not os.path.exists(SOL_DIR):
        os.makedirs(SOL_DIR)

    os.chdir("rust_optimizer")

    for n in range(START_N, END_N + 1):
        in_file = f"{SOL_DIR}/T{n}-2.csv"
        out_file = in_file  # Refinamiento in-place
        
        if not os.path.exists(in_file):
            print(f"⚠️ Saltando T{n}: no existe {in_file}")
            continue
            
        print(f"\n" + "="*60)
        print(f"🔧 REFINANDO T{n}")
        print("="*60)

        # ETAPA 0: Crear base con bloques de 10 (descomentar si es nueva solución)
        run_command([
            BIN_PATH, 
            "--target-n", str(n), 
            "--strategy", "deca", 
            "-o", out_file,
            "--force"
        ])

        # # ETAPA 1: Rim Pressure (Compresión del Borde)
        # print(f"\nETAPA 1: Rim Pressure - Compresión del Borde")
        # run_command([
        #     BIN_PATH, "-i", in_file, "-o", out_file,
        #     "--strategy", "rim_pressure",
        #     "--gravity-steps", "100"
        # ])

        # # ETAPA 2: SA Caliente - Exploración amplia
        # # Temp alta permite escapar de mínimos locales
        print(f"\nETAPA 2: SA Caliente - Exploración amplia")
        run_command([
            BIN_PATH, "-i", out_file, "-o", out_file,
            "--strategy", "sa",
            "--generations", "30000",
            "--sa-init-temp", "0.001",
            "--sa-final-temp", "0.00001",    # ← Importante: final < init
            "--sa-step-scale", "0.1"
        ])

        # # ETAPA 3: Rim Pressure Intermedio
        # print(f"\nETAPA 3: Rim Pressure Intermedio")
        # run_command([
        #     BIN_PATH, "-i", out_file, "-o", out_file,
        #     "--strategy", "rim_pressure",
        #     "--gravity-steps", "50"
        # ])

        # # ETAPA 4: SA Medio - Refinamiento
        # print(f"\nETAPA 4: SA Medio - Refinamiento")
        # run_command([
        #     BIN_PATH, "-i", out_file, "-o", out_file,
        #     "--strategy", "sa",
        #     "--generations", "15000",
        #     "--sa-init-temp", "0.1",
        #     "--sa-final-temp", "0.001",   # ← final < init
        #     "--sa-step-scale", "0.1"
        # ])

        # # ETAPA 5: SA Frío - Pulido fino
        # print(f"\nETAPA 5: SA Frío - Pulido fino")
        run_command([
            BIN_PATH, "-i", out_file, "-o", out_file,
            "--strategy", "sa",
            "--generations", "20000",
            "--sa-init-temp", "0.0001",
            "--sa-final-temp", "0.000001", # ← final < init
            "--sa-step-scale", "0.03"
        ])

        # # ETAPA 6: SA Helado - Microajustes finales
        # print(f"\nETAPA 6: SA Helado - Microajustes finales")
        run_command([
            BIN_PATH, "-i", out_file, "-o", out_file,
            "--strategy", "sa",
            "--generations", "50000",
            "--sa-init-temp", "0.00001",
            "--sa-final-temp", "0.000000001", # ← final < init
            "--sa-step-scale", "0.01"
        ])

        # run_command([
        #     BIN_PATH, "-i", out_file, "-o", out_file,
        #     "--strategy", "sa",
        #     "--generations", "20000",
        #     "--sa-init-temp", "0.0000001",
        #     "--sa-final-temp", "0.0000000001", # ← final < init
        #     "--sa-step-scale", "0.01"
        # ])

        # run_command([
        #     BIN_PATH, "-i", out_file, "-o", out_file,
        #     "--strategy", "sa",
        #     "--generations", "200000",
        #     "--sa-init-temp", "0.01",
        #     "--sa-final-temp", "0.00000001", # ← final < init
        #     "--sa-step-scale", "0.1"
        # ])
        print(f"✅ Finalizado T{n}")
