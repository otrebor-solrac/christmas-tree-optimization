#!/usr/bin/env python3
"""
Optimizador de packing con SIDE FIJO.
Objetivo: Encontrar configuración de N árboles que quepa en un bounding box de lado fijo.
Cada árbol tiene 3 parámetros optimizables: (x, y, angle).
"""
import cma
import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
from shapely.strtree import STRtree
import math

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
TREE_HEIGHT = 1.0  # Altura aproximada del árbol

def create_tree_poly(x, y, angle):
    """Crea un polígono de árbol en posición (x, y) con ángulo dado."""
    rotated = rotate(BASE_POLY, angle, origin=(0, 0))
    return translate(rotated, x, y)

def check_kaggle_collision(polys):
    """Chequeo de colisión exacto de Kaggle."""
    if not polys: return False, 0
    tree = STRtree(polys)
    collision_count = 0
    for i, p in enumerate(polys):
        indices = tree.query(p)
        for idx in indices:
            if idx <= i: continue
            if p.intersects(polys[idx]) and not p.touches(polys[idx]):
                collision_count += 1
    return collision_count > 0, collision_count

def make_objective(n_trees, target_side):
    """
    Cada árbol tiene 3 parámetros: (x, y, angle)
    Total de parámetros: n_trees * 3
    
    Penaliza:
    - Colisiones entre árboles
    - Árboles fuera del bounding box objetivo
    """
    half_side = target_side / 2.0
    target_box = box(-half_side, -half_side, half_side, half_side)
    
    def objective_function(params):
        # Decodificar parámetros: [x1, y1, a1, x2, y2, a2, ...]
        polys = []
        for i in range(n_trees):
            x = params[i * 3]
            y = params[i * 3 + 1]
            angle = params[i * 3 + 2]
            polys.append(create_tree_poly(x, y, angle))
        
        # Penalización por colisiones
        has_collision, collision_count = check_kaggle_collision(polys)
        collision_penalty = collision_count * 10.0
        
        # Penalización por árboles fuera del box
        out_of_bounds_penalty = 0.0
        for p in polys:
            if not target_box.contains(p):
                # Calcular cuánto sale del box
                diff = p.difference(target_box)
                out_of_bounds_penalty += diff.area * 100.0
        
        # Si hay colisiones o fuera de límites, penalizar fuertemente
        if collision_penalty > 0 or out_of_bounds_penalty > 0:
            return 50.0 + collision_penalty + out_of_bounds_penalty
        
        # Si es válido, retornar el side real (debería ser <= target_side)
        minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
        for p in polys:
            b = p.bounds
            if b[0] < minx: minx = b[0]
            if b[1] < miny: miny = b[1]
            if b[2] > maxx: maxx = b[2]
            if b[3] > maxy: maxy = b[3]
        
        real_side = max(maxx - minx, maxy - miny)
        return real_side
    
    return objective_function

def generate_initial_grid(n_trees, target_side):
    """
    Genera una configuración inicial basada en los ángulos encontrados para T70.
    Alterna entre angle_up (~14.73°) y angle_down (~194.73°).
    """
    half_side = target_side / 2.0 - 0.5
    cols = int(math.ceil(math.sqrt(n_trees * 1.5)))
    rows = int(math.ceil(n_trees / cols))
    
    spacing_x = (2 * half_side) / max(cols - 1, 1)
    spacing_y = (2 * half_side) / max(rows - 1, 1)
    
    # Ángulos encontrados en T70
    ANGLE_UP = 14.73
    ANGLE_DOWN = 194.73
    
    params = []
    for i in range(n_trees):
        col = i % cols
        row = i // cols
        x = -half_side + col * spacing_x
        y = -half_side + row * spacing_y
        # Alternar ángulos como en zip_skew
        angle = ANGLE_UP if i % 2 == 0 else ANGLE_DOWN
        params.extend([x, y, angle])
    
    return params

def optimize_fixed_side(n_trees, target_side, output_path, iterations=500, popsize=50):
    """
    Optimiza N árboles dentro de un bounding box de lado fijo.
    Ángulos limitados a ±3° alrededor de los valores óptimos encontrados.
    """
    print(f"🚀 Optimizando N={n_trees} con side fijo = {target_side}")
    print(f"   📐 Dimensiones: {n_trees * 3} parámetros ({n_trees} × 3)")
    print(f"   🎯 Ángulos limitados: 14.73° ± 3° y 194.73° ± 3°")
    
    half_side = target_side / 2.0
    
    # Ángulos base encontrados
    ANGLE_UP = 14.73
    ANGLE_DOWN = 194.73
    ANGLE_TOLERANCE = 3.0
    
    # Bounds para cada árbol
    margin = 0.4
    bounds_min = []
    bounds_max = []
    for i in range(n_trees):
        bounds_min.extend([-half_side + margin, -half_side + margin, 0])
        bounds_max.extend([half_side - margin, half_side - margin, 0])
        # Determinar rango de ángulo según paridad (up/down alternando)
        if i % 2 == 0:
            bounds_min[-1] = ANGLE_UP - ANGLE_TOLERANCE
            bounds_max[-1] = ANGLE_UP + ANGLE_TOLERANCE
        else:
            bounds_min[-1] = ANGLE_DOWN - ANGLE_TOLERANCE
            bounds_max[-1] = ANGLE_DOWN + ANGLE_TOLERANCE
    
    bounds = [bounds_min, bounds_max]
    
    # Configuración inicial
    x0 = generate_initial_grid(n_trees, target_side)
    
    objective = make_objective(n_trees, target_side)
    
    print("   Fase 1: Exploración...")
    es = cma.CMAEvolutionStrategy(x0, 0.5, {
        'bounds': bounds,
        'popsize': popsize,
        'maxiter': iterations,
        'verbose': -9
    })
    es.optimize(objective)
    
    best_score = es.result.fbest
    best_params = es.result.xbest
    print(f"   Mejor score: {best_score:.6f}")
    
    if best_score < 50:  # Solución válida
        # Verificar y guardar
        polys = []
        for i in range(n_trees):
            x = best_params[i * 3]
            y = best_params[i * 3 + 1]
            angle = best_params[i * 3 + 2]
            polys.append(create_tree_poly(x, y, angle))
        
        has_collision, _ = check_kaggle_collision(polys)
        
        if not has_collision:
            minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
            for p in polys:
                b = p.bounds
                if b[0] < minx: minx = b[0]
                if b[1] < miny: miny = b[1]
                if b[2] > maxx: maxx = b[2]
                if b[3] > maxy: maxy = b[3]
            
            real_side = max(maxx - minx, maxy - miny)
            kaggle_score = (real_side ** 2) / n_trees
            
            print(f"\n✅ VÁLIDO! Score Kaggle: {kaggle_score:.6f} (Lado: {real_side:.4f})")
            save_solution(n_trees, best_params, output_path, kaggle_score)
            return kaggle_score
        else:
            print("❌ Solución final tiene colisiones")
    else:
        print(f"❌ No se encontró solución válida (score={best_score:.2f})")
    
    return None

def save_solution(n_trees, params, path, score):
    """Guarda la solución en formato CSV."""
    lines = ["id,x,y,deg,score"]
    for i in range(n_trees):
        x = params[i * 3]
        y = params[i * 3 + 1]
        angle = params[i * 3 + 2]
        lines.append(f"{i+1},s{x:.16f},s{y:.16f},s{angle:.16f},s{score:.18f}")
    
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"   💾 Guardado en {path}")

if __name__ == "__main__":
    N = 70
    TARGET_SIDE = 4.9464  # Side del competidor a igualar/superar
    OUTPUT = "solutions/T70_fixed.csv"
    
    print(f"=" * 60)
    print(f"  FIXED SIDE OPTIMIZER: N={N} | Target={TARGET_SIDE}")
    print(f"=" * 60)
    
    # Con 70 árboles y 3 parámetros cada uno = 210 dimensiones
    # Necesitamos más población e iteraciones
    optimize_fixed_side(N, TARGET_SIDE, OUTPUT, iterations=1000, popsize=100)
