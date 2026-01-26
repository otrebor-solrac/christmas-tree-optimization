import cma
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
import math

# --- CONFIGURACIÓN N=47 (8x4) ---
N_TREES = 33
COLS_OF_PAIRS = 6   # 2 * COLS_OF_PAIRS * ROWS = N_TREES
ROWS = 6            # sqrt(47/4) ≈ 3.4, redondeamos a 4

# --- GEOMETRÍA ---
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
EXPECTED_AREA = BASE_POLY.area * N_TREES

# --- GENERADOR DE RETÍCULA ---
def generate_skewed_lattice(params):
    angle_base = params[0]
    pair_dx    = params[1]
    pair_dy    = params[2]
    col_step   = params[3]
    row_skew   = params[4]
    row_height = params[5]
    
    polys = []
    
    # REGLA: Down = Up + 180
    angle_up   = angle_base
    angle_down = angle_base + 180.0
    
    p_up   = rotate(BASE_POLY, angle_up, origin=(0,0))
    p_down = rotate(BASE_POLY, angle_down, origin=(0,0))
    
    count = 0
    for r in range(ROWS):
        for c in range(COLS_OF_PAIRS):
            if count >= N_TREES: break
            
            # Fórmula diagonal
            bx = (c * col_step) + (r * row_skew)
            by = (r * row_height)
            
            # Árbol 1
            polys.append(translate(p_up, bx, by))
            count += 1
            
            # Árbol 2
            if count < N_TREES:
                polys.append(translate(p_down, bx + pair_dx, by + pair_dy))
                count += 1
                
    return polys

# --- COST FUNCTION ---
def objective_function(params):
    polys = generate_skewed_lattice(params)
    
    try:
        u = unary_union(polys)
    except:
        return 9999.0 
        
    overlap = EXPECTED_AREA - u.area
    
    if overlap > 1e-6:
        # Penalización fuerte si hay superposición
        return 100 + (overlap * 100000)
    
    # Calcular Score (Caja Rotada Mínima)
    rect = u.minimum_rotated_rectangle
    x, y = rect.exterior.coords.xy
    side_a = math.sqrt((x[1]-x[0])**2 + (y[1]-y[0])**2)
    side_b = math.sqrt((x[2]-x[1])**2 + (y[2]-y[1])**2)
    
    return max(side_a, side_b)

# --- EJECUCIÓN ---
if __name__ == "__main__":
    print("🌲 INICIANDO OPTIMIZACIÓN (RANGO 0° - 45°)...")
    
    # VALORES INICIALES (x0)
    x0 = [
        5.0,   # angle_base
        0.20,   # pair_dx
        0.60,   # pair_dy
        0.90,   # col_step
        0.40,   # row_skew
        1.10    # row_height
    ]
    
    # El primer par [0, 45] permite bajar hasta 0 grados.
    bounds = [
        [ 0,   -0.7,  0.0,  0.0,  0.0,  0.0], # Mínimos (Ángulo min = 0)
        [ 12,   0.7,  1.0,  1.5,  1.5,  1.5]  # Máximos
    ]
    
    opts = {
        'popsize': 100,
        'maxiter': 500,
        'bounds': bounds,
        'tolfun': 1e-5,
        'verbose': 1
    }
    
    es = cma.CMAEvolutionStrategy(x0, 0.1, opts)
    es.optimize(objective_function)
    
    best = es.result.xbest
    score = es.result.fbest
    
    print("\n" + "="*60)
    print(f"RESULTADO: {score:.6f}")
    print("="*60)
    print(f"Angle Base:   {best[0]:.4f}")
    print(f"Angle Down:   {best[0] + 180:.4f}")
    print(f"Pair DX:      {best[1]:.4f}")
    print(f"Pair DY:      {best[2]:.4f}")
    print(f"Col Step:     {best[3]:.4f}")    
    print(f"Row Skew:     {best[4]:.4f}")
    print(f"Row Height:   {best[5]:.4f}")

    
    # --- CSV GENERATOR ---
    print("\n>>> CSV FINAL:")
    print("id,x,y,deg,score")
    
    polys = generate_skewed_lattice(best)
    u = unary_union(polys)
    rect = u.minimum_rotated_rectangle
    x, y = rect.exterior.coords.xy
    global_angle_deg = math.degrees(math.atan2(y[1]-y[0], x[1]-x[0]))
    center = u.centroid
    rad = math.radians(-global_angle_deg)
    
    idx = 1
    angle_up = best[0]
    angle_down = best[0] + 180
    
    count = 0
    for r in range(ROWS):
        for c in range(COLS_OF_PAIRS):
            if count >= N_TREES: break
            
            bx = (c * best[3]) + (r * best[4])
            by = (r * best[5])
            
            # T1
            dx, dy = bx - center.x, by - center.y
            nx = center.x + dx*math.cos(rad) - dy*math.sin(rad)
            ny = center.y + dx*math.sin(rad) + dy*math.cos(rad)
            print(f"{idx},s{nx:.16f},s{ny:.16f},s{angle_up - global_angle_deg:.16f},0.43")
            idx += 1; count += 1
            
            # T2
            if count < N_TREES:
                tx = bx + best[1]; ty = by + best[2]
                dx, dy = tx - center.x, ty - center.y
                nx = center.x + dx*math.cos(rad) - dy*math.sin(rad)
                ny = center.y + dx*math.sin(rad) + dy*math.cos(rad)
                print(f"{idx},s{nx:.16f},s{ny:.16f},s{angle_down - global_angle_deg:.16f},0.43")
                idx += 1; count += 1