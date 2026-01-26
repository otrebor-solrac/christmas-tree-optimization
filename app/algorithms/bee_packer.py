#!/usr/bin/env python3
"""
🐝 Artificial Bee Colony (ABC) Packer 🐝
----------------------------------------
Un algoritmo de inteligencia de enjambre diseñado para optimización continua.
Diferente a SA y CMA-ES.
- Abejas Empleadas: Refinan soluciones existentes.
- Abejas Observadoras: Se enfocan en las soluciones prometedoras.
- Abejas Exploradoras (Scouts): Reinician soluciones estancadas (anti-trampas locales).
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
from shapely.strtree import STRtree
import argparse
import os
import time
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

# --- CLASE SOLUCIÓN (FOOD SOURCE) ---
class Solution:
    def __init__(self, n_trees, bounds_range=10.0, init_params=None):
        self.n = n_trees
        if init_params is not None:
            self.params = np.array(init_params, dtype=float)
        else:
            # Generación aleatoria inicial: [x1, y1, a1, x2, y2, a2...]
            # Inicializamos en un cuadrado pequeño proporcional a sqrt(N)
            side = np.sqrt(n_trees) * 1.5
            self.params = (np.random.rand(n_trees * 3) - 0.5)
            # Escalar posiciones
            self.params[0::3] *= side  # x
            self.params[1::3] *= side  # y
            self.params[2::3] *= 360.0 # angle
            
        self.cost = float('inf')
        self.fitness = 0.0
        self.trial = 0 # Contador de estancamiento
        self.evaluate()

    def evaluate(self):
        polys = []
        for i in range(0, len(self.params), 3):
            p = translate(rotate(BASE_POLY, self.params[i+2], origin=(0,0)), 
                          self.params[i], self.params[i+1])
            polys.append(p)
            
        # 1. Bounding Box
        minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
        for p in polys:
            b = p.bounds
            minx = min(minx, b[0]); miny = min(miny, b[1])
            maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
        
        side = max(maxx - minx, maxy - miny)
        
        # 2. Colisiones (Área de Solapamiento Contínua para Gradiente)
        tree = STRtree(polys)
        overlap_area = 0.0
        
        for i, p in enumerate(polys):
            indices = tree.query(p)
            for idx in indices:
                if idx > i: # Evitar duplicados
                    if p.intersects(polys[idx]): 
                        # Calculamos el área exacta para dar un gradiente suave
                        overlap_area += p.intersection(polys[idx]).area
                        
        # 3. Gravedad (Presión hacia el centro)
        # Ayuda a compactar la solución de forma natural
        gravity = np.sum(np.sqrt(self.params[0::3]**2 + self.params[1::3]**2)) / self.n * 0.05

        # Función de Costo Total
        # Penalizamos fuertemente el área de solapamiento
        self.cost = side + (overlap_area * 1000.0) + gravity
        
        # Fitness para ABC (debe ser maximizable y no negativo)
        if self.cost >= 0:
            self.fitness = 1.0 / (1.0 + self.cost)
        else:
            self.fitness = 1.0 + abs(self.cost)

# --- ALGORITMO ABC ---
def run_abc(n, output_path, max_iters=2000, colony_size=40, limit=100):
    """
    colony_size: Número total de abejas (mitad empleadas, mitad observadoras)
    limit: Número de intentos fallidos antes de abandonar una fuente (Scout bee)
    """
    n_employed = colony_size // 2
    
    print(f"🐝 Iniciando ABC | N={n} | Pob={colony_size} | Limit={limit}")
    
    # 1. Inicialización
    population = [Solution(n) for _ in range(n_employed)]
    best_sol = min(population, key=lambda x: x.cost)
    
    start_time = time.time()
    
    for iteration in range(max_iters):
        # --- FASE ABEJAS EMPLEADAS ---
        for i in range(n_employed):
            # Mutar solución i (Guiada por la mejor solución)
            new_sol = mutate(population[i], population, i, best_sol)
            
            # Selección Greedy
            if new_sol.fitness > population[i].fitness:
                population[i] = new_sol
                population[i].trial = 0
            else:
                population[i].trial += 1
                
        # --- FASE ABEJAS OBSERVADORAS (Onlooker) ---
        total_fitness = sum(s.fitness for s in population)
        probs = [s.fitness / total_fitness for s in population]
        
        for _ in range(n_employed):
            i = np.random.choice(range(n_employed), p=probs)
            
            # Mutar solución elegida (G-Best)
            new_sol = mutate(population[i], population, i, best_sol)
            
            if new_sol.fitness > population[i].fitness:
                population[i] = new_sol
                population[i].trial = 0
            else:
                population[i].trial += 1
        
        # --- FASE ABEJAS EXPLORADORAS (Scout) ---
        for i in range(n_employed):
            if population[i].trial > limit:
                # Abandonar y crear una basada en el récord (en lugar de totalmente azar)
                # Esto es una mutación fuerte del récord para no perder la estructura buena
                new_params = np.copy(best_sol.params)
                # Mutar el 50% de los árboles aleatoriamente
                for t_idx in range(n):
                    if random.random() < 0.5:
                        new_params[t_idx*3] += (random.random() - 0.5) * 2.0
                        new_params[t_idx*3 + 1] += (random.random() - 0.5) * 2.0
                
                population[i] = Solution(n, init_params=new_params)
                population[i].trial = 0

        # --- MEMORIZAR MEJOR ---
        current_best = min(population, key=lambda x: x.cost)
        if current_best.cost < best_sol.cost:
            best_sol = copy.deepcopy(current_best)
            if best_sol.cost < 100:
                print(f"   ✨ Iter {iteration}: Nuevo Mejor Coste = {best_sol.cost:.6f} | Overlap: {(best_sol.cost - best_sol.evaluate_side()):.4f}")

        if iteration % 100 == 0:
            print(f"   Iter {iteration}/{max_iters} | Best: {best_sol.cost:.4f} | Avg Fit: {np.mean([p.fitness for p in population]):.2e}")

    # --- RESULTADO FINAL ---
    print(f"\n🏆 ABC Terminado. Mejor Coste: {best_sol.cost:.6f}")
    
    # Guardar CSV
    # Recalcular score real (geometría exacta)
    polys = []
    params = best_sol.params
    for i in range(0, len(params), 3):
        polys.append(translate(rotate(BASE_POLY, params[i+2], origin=(0,0)), params[i], params[i+1]))
        
    minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
    for p in polys:
        b = p.bounds
        minx = min(minx, b[0]); miny = min(miny, b[1])
        maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
    
    final_side = max(maxx - minx, maxy - miny)
    kaggle_score = (final_side ** 2) / n
    
    print(f"📦 Side Real: {final_side:.6f}")
    print(f"📊 Kaggle Score: {kaggle_score:.6f}")
    
    lines = ["id,x,y,deg,score"]
    idx = 1
    for i in range(0, len(params), 3):
        lines.append(f"{idx},s{params[i]:.16f},s{params[i+1]:.16f},s{params[i+2]:.16f},s{final_side:.16f}")
        idx += 1
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

def mutate(current_sol, population, idx, best_sol):
    """
    Ecuación de mutación G-Best: v_ij = x_ij + phi * (x_ij - x_kj) + psi * (best_j - x_ij)
    """
    new_params = np.copy(current_sol.params)
    
    neighbor_idx = idx
    while neighbor_idx == idx:
        neighbor_idx = np.random.randint(0, len(population))
    neighbor_params = population[neighbor_idx].params
    
    # Probabilidad de mutar cada árbol (Multi-Variable Mutation)
    # Entre un 10% y 30% de los árboles cambian por vez
    prob_mut = random.uniform(0.1, 0.3)
    
    for tree_idx in range(current_sol.n):
        if random.random() < prob_mut:
            indices = [tree_idx*3, tree_idx*3+1, tree_idx*3+2]
            
            for j in indices:
                phi = (random.random() - 0.5) * 2.0 # Exploración
                psi = random.random() * 1.5         # Atracción al récord (G-Best)
                
                diff_neighbor = current_sol.params[j] - neighbor_params[j]
                diff_best = best_sol.params[j] - current_sol.params[j]
                
                new_params[j] = current_sol.params[j] + phi * diff_neighbor + psi * diff_best
        
    return Solution(current_sol.n, init_params=new_params)

# Añadimos un helper a Solution para el log
if not hasattr(Solution, 'evaluate_side'):
    def evaluate_side(self):
        polys = []
        for i in range(0, len(self.params), 3):
            p = translate(rotate(BASE_POLY, self.params[i+2], origin=(0,0)), 
                          self.params[i], self.params[i+1])
            polys.append(p)
        minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
        for p in polys:
            b = p.bounds
            minx = min(minx, b[0]); miny = min(miny, b[1])
            maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
        return max(maxx - minx, maxy - miny)
    Solution.evaluate_side = evaluate_side

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--iters', type=int, default=3000)
    args = parser.parse_args()
    
    run_abc(args.n, args.output, max_iters=args.iters)