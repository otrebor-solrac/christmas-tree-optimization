#!/usr/bin/env python3
"""
🦅 PSO Packer (Particle Swarm Optimization) 🦅
----------------------------------------------
Alternativa a SA y CMA-ES.
Usa una 'bandada' de soluciones que vuelan por el espacio de búsqueda.
Ideal para escapar de mínimos locales donde SA se atasca.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from shapely.strtree import STRtree
import argparse
import os
import time
import random

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

# --- PARTÍCULA PSO ---
class Particle:
    def __init__(self, n_trees, bounds_size):
        self.n = n_trees
        # Posición: [x1, y1, a1, x2, y2, a2...]
        # Inicialización aleatoria en un área razonable
        self.position = (np.random.rand(n_trees * 3) - 0.5)
        self.position[0::3] *= bounds_size # x
        self.position[1::3] *= bounds_size # y
        self.position[2::3] *= 360.0       # angle
        
        # Velocidad inicial
        self.velocity = (np.random.rand(n_trees * 3) - 0.5) * 0.5
        
        # Mejor posición personal (pBest)
        self.pbest_position = np.copy(self.position)
        self.pbest_score = float('inf')
        
        self.current_score = float('inf')

    def evaluate(self):
        polys = []
        for i in range(0, len(self.position), 3):
            # Crear polígono geométrico
            p = translate(rotate(BASE_POLY, self.position[i+2], origin=(0,0)), 
                          self.position[i], self.position[i+1])
            polys.append(p)
            
        # 1. Bounding Box
        minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
        for p in polys:
            b = p.bounds
            minx = min(minx, b[0]); miny = min(miny, b[1])
            maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
        
        side = max(maxx - minx, maxy - miny)
        
        # 2. Área de Solapamiento (Gradiente Continuo)
        tree = STRtree(polys)
        overlap_area = 0.0
        
        for i, p in enumerate(polys):
            indices = tree.query(p)
            for idx in indices:
                if idx > i: # Evitar duplicados
                    if p.intersects(polys[idx]): 
                        # Área de intersección para dar gradiente al PSO
                        overlap_area += p.intersection(polys[idx]).area
                        
        # 3. Gravedad (Presión hacia el origen)
        # Ayuda a mantener la bandada compacta
        gravity = np.sum(np.sqrt(self.position[0::3]**2 + self.position[1::3]**2)) / self.n * 0.05
            
        self.current_score = side + (overlap_area * 5000.0) + gravity
        
        # Actualizar pBest
        if self.current_score < self.pbest_score:
            self.pbest_score = self.current_score
            self.pbest_position = np.copy(self.position)
            
        return self.current_score

    def get_real_side(self):
        """Calcula el lado real sin penalizaciones para los logs"""
        polys = []
        for i in range(0, len(self.position), 3):
            p = translate(rotate(BASE_POLY, self.position[i+2], origin=(0,0)), 
                          self.position[i], self.position[i+1])
            polys.append(p)
        minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
        for p in polys:
            b = p.bounds
            minx = min(minx, b[0]); miny = min(miny, b[1])
            maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
        return max(maxx - minx, maxy - miny)

# --- ALGORITMO PSO ---
def run_pso(n, output_path, iterations=1000, swarm_size=50):
    print(f"🦅 Iniciando PSO | N={n} | Swarm={swarm_size} | Iters={iterations}")
    
    # Rango inicial: Estimamos lado necesario (sqrt(n)) y damos un poco más de margen
    bounds_size = np.sqrt(n) * 1.5
    
    # Crear enjambre
    swarm = [Particle(n, bounds_size) for _ in range(swarm_size)]
    
    # Global Best (gBest)
    gbest_position = None
    gbest_score = float('inf')
    
    # Hiperparámetros PSO (Inercia dinámica)
    w_start = 0.9
    w_end = 0.4
    c1 = 1.49445 # Cognitivo (tira hacia mi mejor posición)
    c2 = 1.49445 # Social (tira hacia el líder)
    
    # Limitar velocidad máxima (Clamping dinámico proporcional al tamaño)
    v_max = bounds_size * 0.2

    start_time = time.time()
    
    for it in range(iterations):
        # 1. Evaluar Enjambre
        for p in swarm:
            score = p.evaluate()
            if score < gbest_score:
                gbest_score = score
                gbest_position = np.copy(p.position)
                if score < 100: 
                    side = p.get_real_side()
                    overlap = (score - side)
                    print(f"   ✨ Iter {it}: Nuevo Líder = {score:.6f} | Lado: {side:.4f} | Overlap: {overlap:.4f}")
        
        # Inercia con decaimiento lineal
        w = w_start - (w_start - w_end) * (it / iterations)

        # 2. Actualizar Velocidad y Posición
        for p in swarm:
            r1 = np.random.rand(len(p.position))
            r2 = np.random.rand(len(p.position))
            
            # Velocidad: Inercia + Cognitivo + Social
            vel_cognitive = c1 * r1 * (p.pbest_position - p.position)
            vel_social = c2 * r2 * (gbest_position - p.position)
            
            p.velocity = w * p.velocity + vel_cognitive + vel_social
            
            # Clamp velocidad (evitar explosiones)
            p.velocity = np.clip(p.velocity, -v_max, v_max)
            
            # Mover
            p.position += p.velocity
            
        if it % 100 == 0:
            print(f"   Iter {it}/{iterations} | Leader Score: {gbest_score:.4f} (w={w:.3f})")

    print(f"\n🏆 PSO Terminado. Mejor Score: {gbest_score:.6f}")
    
    # Guardar
    # Reconstruir geometría final
    polys = []
    for i in range(0, len(gbest_position), 3):
        polys.append(translate(rotate(BASE_POLY, gbest_position[i+2], origin=(0,0)), 
                       gbest_position[i], gbest_position[i+1]))
        
    minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
    for p in polys:
        b = p.bounds
        minx = min(minx, b[0]); miny = min(miny, b[1])
        maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
    
    real_side = max(maxx - minx, maxy - miny)
    kaggle_score = (real_side ** 2) / n
    
    print(f"📦 Side Real: {real_side:.6f} | Score: {kaggle_score:.6f}")
    
    lines = ["id,x,y,deg,score"]
    idx = 1
    for i in range(0, len(gbest_position), 3):
        lines.append(f"{idx},s{gbest_position[i]:.16f},s{gbest_position[i+1]:.16f},s{gbest_position[i+2]:.16f},s{real_side:.16f}")
        idx += 1
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--swarm', type=int, default=50)
    args = parser.parse_args()
    
    run_pso(args.n, args.output, iterations=args.iters, swarm_size=args.swarm)