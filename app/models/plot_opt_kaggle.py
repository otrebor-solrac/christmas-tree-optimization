import numpy as np
import copy
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union

import csv

def load_data_from_csv(file_path):
    data_list = []
    
    with open(file_path, 'r') as f:
        # Usamos DictReader para mapear automáticamente las columnas por nombre
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                # Limpiamos la 's' si existe y convertimos a float/int
                # .replace('s', '') no hace daño si la 's' no está
                entry = {
                    'id': int(row['id'].replace('s', '')),
                    'x': float(row['x'].replace('s', '')),
                    'y': float(row['y'].replace('s', '')),
                    'deg': float(row['deg'].replace('s', ''))
                }
                data_list.append(entry)
            except ValueError as e:
                print(f"Error procesando fila {row}: {e}")
                continue
                
    return data_list

# --- USO ---
# Reemplaza 'tu_archivo.csv' con la ruta real
csv_path = '/home/rc/workspace/kaggle/ChrismasTree/solutions/T16.csv' 

initial_data = load_data_from_csv(csv_path)

# --- 2. GEOMETRÍA DEL ÁRBOL ---
def get_tree_coords():
    trunk_w, trunk_h = 0.15, 0.2
    base_w, base_y = 0.7, 0.0
    mid_w, tier_2_y = 0.4, 0.25
    top_w, tier_1_y = 0.25, 0.5
    tip_y = 0.8
    trunk_bottom = -trunk_h
    return [
        (0, tip_y), (top_w/2, tier_1_y), (top_w/4, tier_1_y),
        (mid_w/2, tier_2_y), (mid_w/4, tier_2_y),
        (base_w/2, base_y), (trunk_w/2, base_y),
        (trunk_w/2, trunk_bottom), (-trunk_w/2, trunk_bottom),
        (-trunk_w/2, base_y), (-base_w/2, base_y),
        (-mid_w/4, tier_2_y), (-mid_w/2, tier_2_y),
        (-top_w/4, tier_1_y), (-top_w/2, tier_1_y)
    ]

class Tree:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.poly = self.update_poly()
    
    def update_poly(self):
        base = ShapelyPolygon(get_tree_coords())
        r_poly = rotate(base, self.angle, origin=(0,0), use_radians=False)
        self.poly = translate(r_poly, xoff=self.x, yoff=self.y)
        return self.poly

def calculate_score(trees):
    polys = [t.poly for t in trees]
    union = unary_union(polys)
    minx, miny, maxx, maxy = union.bounds
    w = maxx - minx
    h = maxy - miny
    score = (max(w, h)**2) / len(trees)
    return score, w, h

def check_collisions_strict(trees):
    """
    Detección de colisiones ultra-estricta.
    Retorna True si hay CUALQUIER superposición significativa.
    """
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].poly.intersects(trees[j].poly):
                intersection = trees[i].poly.intersection(trees[j].poly)
                # Tolerancia ajustada a 1e-10 para ser muy estricto
                if intersection.area > 1e-10:
                    return True
    return False

# --- 3. ALGORITMO DE MICRO-OPTIMIZACIÓN ---
def optimize_fine_tuning(start_data, iterations=25):
    # Inicializar árboles
    current_trees = [Tree(d['x'], d['y'], d['deg']) for d in start_data]
    
    # 1. VERIFICACIÓN INICIAL
    if check_collisions_strict(current_trees):
        print("⚠️ ALERTA: Los datos iniciales tienen colisión. Intentando reparar...")
        repaired = False
        # Intento de reparación simple (sacudir hasta que sea válido)
        for _ in range(1000):
            temp_trees = copy.deepcopy(current_trees)
            for t in temp_trees:
                t.x += (np.random.random() - 0.5) * 0.01
                t.y += (np.random.random() - 0.5) * 0.01
                t.angle += (np.random.random() - 0.5) * 2.0
                t.poly = t.update_poly()
            
            if not check_collisions_strict(temp_trees):
                current_trees = temp_trees
                print("✅ Datos iniciales reparados. Iniciando optimización.")
                repaired = True
                break
        if not repaired:
            print("❌ No se pudo reparar la configuración inicial. Abortando.")
            return current_trees, 999.0

    # Establecer línea base
    best_trees = copy.deepcopy(current_trees)
    best_score, best_w, best_h = calculate_score(best_trees)
    
    print(f"🔹 Score Base (Válido): {best_score:.6f} | Caja: {best_w:.4f}x{best_h:.4f}")
    
    # Parámetros de recocido (Afinado fino)
    step_pos = 0.05  
    step_rot = 0.1    # Reducido de 15.0 a 0.5 para precisión
    decay = 0.9995    
    
    for i in range(iterations):
        candidate_trees = copy.deepcopy(best_trees)
        
        # Perturbar un árbol
        idx = np.random.randint(0, len(candidate_trees))
        t = candidate_trees[idx]
        
        dx = (np.random.random() - 0.5) * step_pos
        dy = (np.random.random() - 0.5) * step_pos
        d_ang = (np.random.random() - 0.5) * step_rot
        
        t.x += dx
        t.y += dy
        t.angle += d_ang
        t.poly = t.update_poly()
        
        # --- FILTRO CRÍTICO ---
        # Solo procedemos si NO hay colisión
        if not check_collisions_strict(candidate_trees):
            new_score, w, h = calculate_score(candidate_trees)
            
            # Si es válido Y mejora el score, guardamos
            if new_score < best_score:
                best_score = new_score
                best_trees = candidate_trees
                print(f"✨ Iter {i}: Récord Válido! Score: {best_score:.7f} | ({w:.4f}x{h:.4f})")
        
        # Enfriamiento
        step_pos *= decay
        step_rot *= decay
        
        # Re-calentamiento suave si los pasos son muy pequeños
        if step_pos < 0.00001: 
            step_pos = 0.001
            step_rot = 0.1

    return best_trees, best_score

if __name__ == "__main__":
    print("Iniciando Micro-Afinado Estricto (Sin Colisiones)...")
    final_trees, final_score = optimize_fine_tuning(initial_data, iterations=25000)
    
    print("\n" + "="*40)
    print(f"🏆 RESULTADO FINAL VÁLIDO: {final_score:.6f}")
    print("="*40)
    
    print("id,x,y,deg,score")
    for i, t in enumerate(final_trees):
        # Agregamos la 's' si es necesaria para tu formato final, o la quitamos si no.
        # Aquí pongo formato estándar limpio para CSV numérico.
        print(f"{i+1},s{t.x},s{t.y},s{t.angle},s{final_score}")