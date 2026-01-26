#!/usr/bin/env python3
"""
🤐 General Zipper Optimizer 🤐
------------------------------
Supera las limitaciones de ZipSkew y Honeycomb.
1. Permite anchos IMPARES (ej. 5 columnas) con patrón intercalado (Up/Down).
2. Permite 'ZigZag' vertical para encajar filas.
3. Incluye Grid Search y Rotación Global.
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

# --- 2. GENERADOR FLEXIBLE ---
def generate_lattice(params, n_trees, fixed_cols):
    # params: [angle, dx, dy, row_offset, col_offset_y, global_rot]
    angle      = params[0]
    dx         = params[1]
    dy         = params[2]
    row_offset = params[3] # Desplazamiento X de filas alternas
    col_off_y  = params[4] # Desplazamiento Y de columnas alternas (ZigZag vertical)
    global_rot = params[5]
    
    rows = int(math.ceil(n_trees / float(fixed_cols)))
    
    # Pre-calcular orientaciones
    p_up   = rotate(BASE_POLY, angle, origin=(0,0))
    p_down = rotate(BASE_POLY, angle + 180.0, origin=(0,0))
    
    lattice = []
    count = 0
    
    # Centrado
    grid_w = (fixed_cols - 1) * dx
    grid_h = (rows - 1) * dy
    off_x, off_y = -grid_w/2.0, -grid_h/2.0
    
    for r in range(rows):
        # Offset X para filas pares (opcional, el optimizador lo usará si sirve)
        current_row_x = 0.0 if (r % 2 == 0) else row_offset
        
        for c in range(fixed_cols):
            if count >= n_trees: break
            
            # Patrón Zipper: Columnas pares UP, impares DOWN
            # (O viceversa, el ángulo base decide)
            poly = p_up if (c % 2 == 0) else p_down
            
            # Offset Y para columnas impares (ZigZag vertical)
            # Esto permite que los 'Down' bajen un poco más que los 'Up'
            current_col_y = 0.0 if (c % 2 == 0) else col_off_y
            
            bx = (c * dx) + current_row_x + off_x
            by = (r * dy) + current_col_y + off_y
            
            lattice.append(translate(poly, bx, by))
            count += 1
            
    if abs(global_rot) > 1e-5:
        return [rotate(p, global_rot, origin=(0,0)) for p in lattice], rows
    return lattice, rows

# --- 3. FUNCIÓN OBJETIVO ---
def make_objective(n_trees, fixed_cols):
    def objective(params):
        polys, _ = generate_lattice(params, n_trees, fixed_cols)
        
        # Bounding Box
        minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
        for p in polys:
            b = p.bounds
            minx = min(minx, b[0]); miny = min(miny, b[1])
            maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
            
        w, h = maxx - minx, maxy - miny
        max_side = max(w, h)
        
        # Colisiones
        tree = STRtree(polys)
        collision = False
        for i, p in enumerate(polys):
            idx_list = tree.query(p)
            for j in idx_list:
                if j > i and p.intersects(polys[j]) and not p.touches(polys[j]):
                    collision = True; break
            if collision: break
        
        # Penalización
        return max_side + (1000.0 if collision else 0.0)
    return objective

# --- 4. OPTIMIZADOR PRINCIPAL ---
def optimize_n(n, output_path):
    print(f"🤐 General Zipper | N={n} | Buscando grids flexibles (Pares e Impares)...")
    
    # Rango amplio de columnas
    base_cols = int(np.ceil(np.sqrt(n)))
    # Probamos desde la mitad hasta el doble, el 'Grid Search' es barato
    candidates = range(max(1, base_cols - 3), base_cols + 5)
    
    best_score = float('inf')
    best_res = None
    best_cols = 0
    
    for cols in candidates:
        if cols < 1: continue
        rows = int(math.ceil(n / float(cols)))
        if rows < 1: continue
        
        # [angle, dx, dy, row_offset, col_off_y, global_rot]
        # dx ~ 0.4 (solapamiento fuerte), dy ~ 1.0
        x0 = [0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
        
        # Bounds generosos
        bounds = [
            [-180, 0.1, 0.4, -1.0, -1.0, -45], 
            [ 180, 1.0, 1.5,  1.0,  1.0,  45]
        ]
        
        # Optimización rápida
        opts = {'popsize': 32, 'maxiter': 150, 'bounds': bounds, 'verbose': -9}
        es = cma.CMAEvolutionStrategy(x0, 0.5, opts)
        
        try:
            es.optimize(make_objective(n, cols))
        except: continue
        
        score = es.result.fbest
        print(f"   Grid {cols}x{rows} -> Score: {score:.4f}")
        
        if score < best_score:
            best_score = score
            best_res = es.result.xbest
            best_cols = cols
            
    if best_res is None: return False
    
    print(f"🏆 Ganador: Grid {best_cols} cols. Refinando...")
    
    # Refinar
    x0 = best_res
    bounds = [[-180, 0.1, 0.4, -1.0, -1.0, -45], [180, 1.0, 1.5, 1.0, 1.0, 45]]
    opts = {'popsize': 48, 'maxiter': 400, 'bounds': bounds, 'verbose': -9}
    es = cma.CMAEvolutionStrategy(x0, 0.2, opts)
    es.optimize(make_objective(n, best_cols))
    
    final_params = es.result.xbest
    final_polys, final_rows = generate_lattice(final_params, n, best_cols)
    
    # Métricas finales
    minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
    for p in final_polys:
        b = p.bounds
        minx = min(minx, b[0]); miny = min(miny, b[1])
        maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
    
    w, h = maxx - minx, maxy - miny
    max_side = max(w, h)
    final_score = (max_side ** 2) / n
    
    print(f"✨ Final: Side={max_side:.4f} | Score={final_score:.6f} | Grid={best_cols}x{final_rows}")
    
    # Guardar CSV
    angle      = final_params[0]
    dx         = final_params[1]
    dy         = final_params[2]
    row_offset = final_params[3]
    col_off_y  = final_params[4]
    global_rot_rad = math.radians(final_params[5])
    
    grid_w = (best_cols - 1) * dx
    grid_h = (final_rows - 1) * dy
    off_x, off_y = -grid_w/2.0, -grid_h/2.0
    
    cos_g = math.cos(global_rot_rad)
    sin_g = math.sin(global_rot_rad)
    
    lines = ["id,x,y,deg,score"]
    idx = 1
    count = 0
    
    for r in range(final_rows):
        cur_row_x = 0.0 if r%2==0 else row_offset
        for c in range(best_cols):
            if count >= n: break
            
            base_angle = angle if c%2==0 else angle+180
            cur_col_y = 0.0 if c%2==0 else col_off_y
            
            bx = (c * dx) + cur_row_x + off_x
            by = (r * dy) + cur_col_y + off_y
            
            # Rotar
            rx = bx * cos_g - by * sin_g
            ry = bx * sin_g + by * cos_g
            fa = base_angle + final_params[5]
            
            lines.append(f"{idx},s{rx:.16f},s{ry:.16f},s{fa:.16f},s{max_side:.16f}")
            idx+=1; count+=1
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=25)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    
    out = args.output or f"solutions/T{args.n}.csv"
    optimize_n(args.n, out)