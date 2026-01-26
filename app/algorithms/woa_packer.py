#!/usr/bin/env python3
"""
🐋 Hybrid Whale Optimization Algorithm (HWOA) 🐋
------------------------------------------------
Combina la búsqueda espiral de las ballenas (WOA) con
la capacidad de escape del Recocido Simulado (SA).
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from shapely.strtree import STRtree
import argparse
import os
import math
import random
import copy

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

# --- CLASE BALLENA (SOLUCIÓN) ---
class Whale:
    def __init__(self, n_trees, bounds_size):
        self.n = n_trees
        # Vector de posición: [x1, y1, a1, x2, y2, a2...]
        self.position = (np.random.rand(n_trees * 3) - 0.5)
        self.position[0::3] *= bounds_size # x
        self.position[1::3] *= bounds_size # y
        self.position[2::3] *= 360.0       # angle
        self.cost = float('inf')
        
    def evaluate(self):
        polys = []
        for i in range(0, len(self.position), 3):
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
        
        # 2. Colisiones
        tree = STRtree(polys)
        collisions = 0
        for i, p in enumerate(polys):
            indices = tree.query(p)
            for idx in indices:
                if idx > i and p.intersects(polys[idx]):
                    collisions += 1
        
        # Penalización
        penalty = 0
        if collisions > 0:
            penalty = collisions * 1000.0 + 100.0
            
        self.cost = side + penalty
        return self.cost

# --- MOTOR SA (Hybrid Component) ---
def local_search_sa(whale, max_steps=50, initial_temp=1.0):
    """Refina una ballena usando Simulated Annealing rápido"""
    current_pos = np.copy(whale.position)
    current_cost = whale.cost
    
    best_pos_sa = np.copy(current_pos)
    best_cost_sa = current_cost
    
    temp = initial_temp
    
    for _ in range(max_steps):
        # Pequeña perturbación
        idx = random.randint(0, len(current_pos)-1)
        old_val = current_pos[idx]
        
        # Mover un valor aleatorio
        shift = (random.random() - 0.5) * temp * 2.0
        current_pos[idx] += shift
        
        # Evaluar (creamos una ballena temporal)
        temp_whale = Whale(whale.n, 1.0) # Bounds da igual aquí
        temp_whale.position = current_pos
        new_cost = temp_whale.evaluate()
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_cost = new_cost
            if new_cost < best_cost_sa:
                best_cost_sa = new_cost
                best_pos_sa = np.copy(current_pos)
        else:
            current_pos[idx] = old_val # Revertir
            
        temp *= 0.90
        
    # Actualizar la ballena original si mejoramos
    if best_cost_sa < whale.cost:
        whale.position = best_pos_sa
        whale.cost = best_cost_sa

# --- ALGORITMO HWOA ---
def run_hwoa(n, output_path, iterations=1000, pop_size=30):
    print(f"🐋 Hybrid WOA | N={n} | Pob={pop_size} | Iters={iterations}")
    
    bounds_size = np.sqrt(n) * 1.5
    population = [Whale(n, bounds_size) for _ in range(pop_size)]
    
    # Evaluar inicial
    for w in population: w.evaluate()
        
    leader = min(population, key=lambda w: w.cost)
    print(f"   Líder Inicial: {leader.cost:.4f}")
    
    for t in range(iterations):
        a = 2.0 - t * (2.0 / iterations) # Decae de 2 a 0 linealmente
        
        for i in range(pop_size):
            r1 = random.random()
            r2 = random.random()
            
            A = 2 * a * r1 - a
            C = 2 * r2
            
            p = random.random()
            
            new_pos = np.copy(population[i].position)
            
            if p < 0.5:
                if abs(A) < 1:
                    # ENCIRCLING (Hacia el líder)
                    D = abs(C * leader.position - population[i].position)
                    new_pos = leader.position - A * D
                else:
                    # SEARCH (Hacia ballena aleatoria)
                    rand_whale = population[random.randint(0, pop_size-1)]
                    D = abs(C * rand_whale.position - population[i].position)
                    new_pos = rand_whale.position - A * D
            else:
                # BUBBLE-NET (Espiral)
                D_leader = abs(leader.position - population[i].position)
                l = (random.random() * 2) - 1
                new_pos = D_leader * math.exp(1 * l) * math.cos(2 * math.pi * l) + leader.position
            
            # Aplicar movimiento
            population[i].position = new_pos
            population[i].evaluate()
            
        # --- HIBRIDACIÓN: Aplicar SA al Líder y a 2 aleatorias ---
        # Esto ayuda a que el líder no se estanque
        local_search_sa(leader, max_steps=100, initial_temp=0.5)
        
        # Actualizar líder global
        current_best = min(population, key=lambda w: w.cost)
        if current_best.cost < leader.cost:
            leader = copy.deepcopy(current_best)
            if leader.cost < 100:
                print(f"   ✨ Iter {t}: Nuevo Líder Híbrido = {leader.cost:.6f}")
                
    # Guardar Resultado
    print(f"🏆 HWOA Terminado. Score: {leader.cost:.6f}")
    
    # Reconstruir para guardar
    polys = []
    params = leader.position
    for i in range(0, len(params), 3):
        polys.append(translate(rotate(BASE_POLY, params[i+2], origin=(0,0)), params[i], params[i+1]))
        
    minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
    for p in polys:
        b = p.bounds
        minx = min(minx, b[0]); miny = min(miny, b[1])
        maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
    
    final_side = max(maxx - minx, maxy - miny)
    
    lines = ["id,x,y,deg,score"]
    idx = 1
    for i in range(0, len(params), 3):
        lines.append(f"{idx},s{params[i]:.16f},s{params[i+1]:.16f},s{params[i+2]:.16f},s{final_side:.16f}")
        idx += 1
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--iters', type=int, default=1000)
    args = parser.parse_args()
    
    run_hwoa(args.n, args.output, iterations=args.iters)