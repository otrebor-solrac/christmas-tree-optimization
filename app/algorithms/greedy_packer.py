#!/usr/bin/env python3
"""
🧮 SciPy Separation Optimizer 🧮
--------------------------------
Usa optimización matemática estricta (L-BFGS-B) para minimizar el solapamiento y el tamaño.
No es heurístico ni aleatorio: sigue el gradiente matemático de la función de costo.

Fases:
1. 'Explosión': Separa los árboles para eliminar el solapamiento masivo inicial.
2. 'Compresión': Aumenta la gravedad al centro manteniendo el solapamiento en 0.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
from scipy.optimize import minimize, basinhopping
import argparse
import os
import time
import math

# --- GEOMETRÍA ---
def get_base_poly():
    # Tu geometría exacta
    trunk_w, trunk_h = 0.15, 0.2
    base_w, base_y = 0.7, 0.0
    mid_w, tier_2_y = 0.4, 0.25
    top_w, tier_1_y = 0.25, 0.5
    tip_y = 0.8
    trunk_bottom = -trunk_h
    coords = [
        (0, tip_y), (top_w/2, tier_1_y), (top_w/4, tier_1_y),
        (mid_w/2, tier_2_y), (mid_w/4, tier_2_y),
        (base_w/2, base_y), (trunk_w/2, base_y),
        (trunk_w/2, trunk_bottom), (-trunk_w/2, trunk_bottom),
        (-trunk_w/2, base_y), (-base_w/2, base_y),
        (-mid_w/4, tier_2_y), (-mid_w/2, tier_2_y),
        (-top_w/4, tier_1_y), (-top_w/2, tier_1_y)
    ]
    return Polygon(coords)

BASE_POLY = get_base_poly()

# Cache para acelerar la creación de polígonos durante la optimización
# (Aunque shapely es rápido, evitar creaciones innecesarias ayuda)
def create_poly(x, y, angle):
    return translate(rotate(BASE_POLY, angle, origin=(0,0)), x, y)

def objective_function(params, n, stage='separation'):
    """
    params: array plano [x1, y1, a1, x2, y2, a2, ...]
    """
    polys = []
    # Reconstruir polígonos
    for i in range(0, len(params), 3):
        polys.append(create_poly(params[i], params[i+1], params[i+2]))

    # 1. Calcular Bounding Box (Métrica Kaggle)
    minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
    for p in polys:
        b = p.bounds
        minx = min(minx, b[0]); miny = min(miny, b[1])
        maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
    
    side = max(maxx - minx, maxy - miny)

    # 2. Calcular Solapamiento (Penalización)
    # Usamos una heurística rápida para el optimizador: 
    # Suma de áreas - Área de la unión.
    # Si no hay solapamiento, esto es 0.
    total_area_sum = BASE_POLY.area * n
    try:
        # unary_union puede ser lento, pero es exacto.
        # Para N=25 es aceptable. Para N=200 puede ser lento.
        union_poly = unary_union(polys)
        union_area = union_poly.area
        overlap = total_area_sum - union_area
    except:
        overlap = 1.0 # Fallback por error geométrico

    # 3. Función de Costo
    if stage == 'separation':
        # Prioridad absoluta: ELIMINAR SOLAPAMIENTO
        # Costo = Overlap * 1000 + Side * 1
        return (overlap * 10000.0) + side
    else: # stage == 'compression'
        # Prioridad: REDUCIR LADO (Manteniendo overlap a raya)
        # Costo = Side * 10 + Overlap * 100000
        return (side * 10.0) + (overlap * 100000.0)

def optimize_scipy(n, output_path):
    print(f"🧮 SciPy Optimizer | N={n}")

    # --- 1. Generación Inicial (Retícula apretada) ---
    # Empezamos con una retícula pequeña para que el solver los "separe"
    cols = int(np.ceil(np.sqrt(n)))
    initial_params = []
    for i in range(n):
        r, c = divmod(i, cols)
        # Espaciado inicial pequeño (0.5) para forzar colisiones y que el solver trabaje
        initial_params.extend([c * 0.5, r * 0.5, 0.0]) 
    
    x0 = np.array(initial_params)

    # --- 2. Fase de Separación (Unoverlap) ---
    print("   🔹 Fase 1: Separación (Eliminando colisiones)...")
    start_t = time.time()
    
    # Usamos Nelder-Mead o Powell porque no tenemos gradiente explícito (shapely no es diferenciable)
    # Powell es robusto sin gradientes.
    res_sep = minimize(
        objective_function, 
        x0, 
        args=(n, 'separation'), 
        method='Powell', 
        options={'disp': True, 'maxiter': 2000, 'ftol': 1e-3}
    )
    
    print(f"      Separación terminada en {time.time()-start_t:.1f}s. Costo: {res_sep.fun:.4f}")
    
    # --- 3. Fase de Compresión ---
    print("   🔹 Fase 2: Compresión (Apretando el bounding box)...")
    
    # Ahora usamos el resultado de la separación como inicio
    x0_comp = res_sep.x
    
    # Podemos intentar COBYLA que maneja restricciones (Constraint Optimization by Linear Approximation)
    # O seguir con Powell/Nelder-Mead ajustando los pesos
    res_comp = minimize(
        objective_function, 
        x0_comp, 
        args=(n, 'compression'), 
        method='Nelder-Mead', # Nelder-Mead es bueno refinando localmente
        options={'disp': True, 'maxiter': 3000, 'xatol': 1e-4}
    )
    
    final_params = res_comp.x
    final_score = objective_function(final_params, n, 'compression') # Nota: devuelve el costo ponderado
    
    # Calcular Score Real (Side)
    polys = []
    for i in range(0, len(final_params), 3):
        polys.append(create_poly(final_params[i], final_params[i+1], final_params[i+2]))
    
    minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
    for p in polys:
        b = p.bounds
        minx = min(minx, b[0]); miny = min(miny, b[1])
        maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
    
    real_side = max(maxx - minx, maxy - miny)
    kaggle_score = (real_side**2) / n
    
    print(f"🏆 Final: Side={real_side:.6f} | Score={kaggle_score:.6f}")

    # Guardar
    lines = ["id,x,y,deg,score"]
    idx = 1
    for i in range(0, len(final_params), 3):
        lines.append(f"{idx},s{final_params[i]:.16f},s{final_params[i+1]:.16f},s{final_params[i+2]:.16f},s{real_side:.16f}")
        idx += 1
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"💾 Guardado en {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    optimize_scipy(args.n, args.output)