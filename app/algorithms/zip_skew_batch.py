#!/usr/bin/env python3
"""
🎄 ZipSkew V4 - Grid Search & Global Rotation 🎄
------------------------------------------------
Diferencia con V3:
En lugar de intentar que CMA-ES "mueva" el número de columnas (que se suele atascar),
esta versión prueba explícitamente 5 o 6 configuraciones de columnas diferentes
y elige la mejor. ¡Infalible!
"""
import cma
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
from shapely.strtree import STRtree
import math
import argparse
import os

# --- 1. GEOMETRÍA ---
def get_base_poly():
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

# --- 2. GENERADOR (Con Columnas Fijas) ---
def generate_skewed_lattice(params, n_trees, fixed_cols):
    # params ya NO incluye cols. Son 7 valores:
    # [angle, dx, dy, col_step, skew, height, ROTACION_GLOBAL]
    angle_base = params[0]
    pair_dx    = params[1]
    pair_dy    = params[2]
    col_step   = params[3]
    row_skew   = params[4]
    row_height = params[5]
    global_rot = params[6]
    
    cols_of_pairs = fixed_cols
    rows = int(math.ceil(n_trees / (2.0 * cols_of_pairs)))
    
    p_up   = rotate(BASE_POLY, angle_base, origin=(0,0))
    p_down = rotate(BASE_POLY, angle_base + 180.0, origin=(0,0))
    
    lattice = []
    count = 0
    
    # Centramos para rotación noble
    grid_w = (cols_of_pairs - 1) * col_step + (rows - 1) * row_skew
    grid_h = (rows - 1) * row_height
    off_x, off_y = -grid_w/2.0, -grid_h/2.0
    
    for r in range(rows):
        for c in range(cols_of_pairs):
            if count >= n_trees: break
            bx = (c * col_step) + (r * row_skew) + off_x
            by = (r * row_height) + off_y
            
            lattice.append(translate(p_up, bx, by))
            count += 1
            if count < n_trees:
                lattice.append(translate(p_down, bx + pair_dx, by + pair_dy))
                count += 1
                
    if abs(global_rot) > 1e-5:
        # Rotamos todos los polígonos
        return [rotate(p, global_rot, origin=(0,0)) for p in lattice], rows
    return lattice, rows

# --- 3. FUNCIÓN OBJETIVO ---
def make_objective(n_trees, fixed_cols):
    def objective(params):
        polys, _ = generate_skewed_lattice(params, n_trees, fixed_cols)
        
        # Bounding Box Rápido
        minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
        for p in polys:
            b = p.bounds
            minx = min(minx, b[0]); miny = min(miny, b[1])
            maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
            
        w, h = maxx - minx, maxy - miny
        max_side = max(w, h)
        
        # Chequeo rápido de colisión (STRtree)
        # Si hay colisión, penalizamos FUERTE para descartar el grid rápido
        tree = STRtree(polys)
        collision = False
        for i, p in enumerate(polys):
            idx_list = tree.query(p)
            for j in idx_list:
                if j > i and p.intersects(polys[j]) and not p.touches(polys[j]):
                    collision = True; break
            if collision: break
            
        if collision:
            # Castigo Suave (Soft Penalty) con gradiente
            # Calculamos el área de superposición real para que el optimizador sepa "cuánto" chocó
            try:
                # El área total sumada es constante para N árboles
                total_area = sum(p.area for p in polys)
                union_area = unary_union(polys).area
                overlap = total_area - union_area
                
                # Penalización: Score Base + (Overlap * Factor)
                # El factor debe ser suficiente para que al final 0 overlap sea mejor
                return max_side + (overlap * 10.0)
            except:
                 return max_side + 100.0 # Fallback si falla shapely
            
        return max_side
    return objective

# --- 4. OPTIMIZADOR PRINCIPAL (GRID SEARCH) ---
def optimize_n(n, output_path, verbose=True):
    if verbose: print(f"🌲 Analizando N={n} con Grid Search...")
    
    # A. Generar candidatos de Grid
    base_cols = int(np.ceil(np.sqrt(n / 4.0))) 
    # Probamos un rango amplio alrededor del estimado
    candidates = range(max(1, base_cols - 2), base_cols + 4)
    
    best_overall_score = float('inf')
    best_overall_params = None
    best_cols = 0
    
    # B. Probar cada candidato rápidamente
    for cols in candidates:
        if cols < 1: continue
        rows = int(math.ceil(n / (2.0 * cols)))
        if rows < 1: continue
        
        # [angle, dx, dy, col_step, skew, height, GLOBAL_ROT]
        x0 = [5.0, 0.3, 0.7, 0.9, 0.3, 1.1, 0.0]
        bounds = [
            [0, -2, -2, 0.5, -3, 0.5, -45], 
            [20, 2,  2, 3.0,  3, 3.0,  45]
        ]
        
        # Optimización rápida (100 iters) para ver si el grid promete
        opts = {'popsize': 50, 'maxiter': 200, 'bounds': bounds, 'verbose': -9}
        es = cma.CMAEvolutionStrategy(x0, 0.5, opts)
        
        try:
            es.optimize(make_objective(n, cols))
        except:
            continue
            
        score = es.result.fbest
        if verbose: print(f"   Grid {cols} cols ({rows} filas) -> Score: {score:.4f}")
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_params = es.result.xbest
            best_cols = cols
            
    # C. Refinar al Ganador
    if best_overall_params is None:
        if verbose: print("   ❌ No se encontró configuración válida.")
        return False
        
    if verbose: print(f"🏆 Ganador: Grid {best_cols} cols. Refinando...")
    
    # Segunda pasada más precisa con el grid ganador
    x0 = best_overall_params
    bounds = [[0,-2,-2,0.5,-3,0.5,-45], [90,2,2,3,3,3,45]]
    opts = {'popsize': 64, 'maxiter': 300, 'bounds': bounds, 'verbose': -9}
    es = cma.CMAEvolutionStrategy(x0, 0.2, opts)
    es.optimize(make_objective(n, best_cols))
    
    final_params = es.result.xbest
    final_score_raw = es.result.fbest
    
    # Calcular métricas finales
    polys, rows = generate_skewed_lattice(final_params, n, best_cols)
    minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
    for p in polys:
        b = p.bounds
        minx = min(minx, b[0]); miny = min(miny, b[1])
        maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
    
    w, h = maxx - minx, maxy - miny
    max_side = max(w, h)
    final_score = (max_side ** 2) / n
    
    if verbose:
        print(f"   ✨ Final: Side={max_side:.6f} | Score={final_score:.6f} | Rot={final_params[6]:.1f}°")

    # D. Guardar CSV
    angle_base = final_params[0]
    pair_dx    = final_params[1]
    pair_dy    = final_params[2]
    col_step   = final_params[3]
    row_skew   = final_params[4]
    row_height = final_params[5]
    global_rot_rad = math.radians(final_params[6])
    
    grid_w = (best_cols - 1) * col_step + (rows - 1) * row_skew
    grid_h = (rows - 1) * row_height
    off_x, off_y = -grid_w/2.0, -grid_h/2.0
    
    cos_g = math.cos(global_rot_rad)
    sin_g = math.sin(global_rot_rad)
    
    lines = ["id,x,y,deg,score"]
    idx = 1
    count = 0
    
    for r in range(rows):
        for c in range(best_cols):
            if count >= n: break
            bx = (c * col_step) + (r * row_skew) + off_x
            by = (r * row_height) + off_y
            
            def write_tree(lx, ly, l_ang):
                rx = lx * cos_g - ly * sin_g
                ry = lx * sin_g + ly * cos_g
                fa = l_ang + final_params[6]
                return f"{idx},s{rx:.16f},s{ry:.16f},s{fa:.16f},s{max_side:.16f}"
            
            lines.append(write_tree(bx, by, angle_base))
            idx += 1; count += 1
            if count < n:
                tx = bx + pair_dx; ty = by + pair_dy
                lines.append(write_tree(tx, ty, angle_base + 180))
                idx += 1; count += 1
                
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int)
    parser.add_argument('--output', type=str)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=200)
    
    args = parser.parse_args()
    
    if args.batch:
        print(f"🚀 Iniciando Batch V4 ({args.start}-{args.end})...")
        for i in range(args.start, args.end + 1):
            out = f"solutions/T{i}.csv"
            optimize_n(i, out)
    elif args.n:
        out = args.output or f"solutions/T{args.n}.csv"
        optimize_n(args.n, out)
    else:
        print("Uso: python zip_skew_v4.py --n 47")