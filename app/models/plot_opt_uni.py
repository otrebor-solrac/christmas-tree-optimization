import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union

# ==========================================
# 1. INPUT: PEGA AQUÍ TU RESULTADO ANTERIOR
# ==========================================
RAW_INPUT_DATA = """

"""

# --- CONFIGURACIÓN ---
PLOT_EVERY = 20    # Cada cuántos intentos actualiza la gráfica
PAUSE_TIME = 0.001 # Velocidad de animación

# --- PARSEO DE DATOS (LA NUEVA FUNCIÓN) ---
def parse_input_to_initial_data(text):
    data_list = []
    text = text.strip()
    
    # Si no hay texto, devolvemos None para usar defaults
    if not text:
        return None

    lines = text.split('\n')
    for line in lines:
        # Ignorar líneas vacías o la cabecera de FINAL L
        if not line.strip() or "FINAL" in line:
            continue
            
        try:
            parts = line.split(',')
            # Formato esperado: ID, sX, sY, sAngle, sL
            # Limpiamos la 's' y convertimos
            p_id = int(parts[0])
            x = float(parts[1].replace('s', ''))
            y = float(parts[2].replace('s', ''))
            deg = float(parts[3].replace('s', ''))
            
            data_list.append({'id': p_id, 'x': x, 'y': y, 'deg': deg})
        except Exception as e:
            print(f"Saltando línea inválida: {line} ({e})")
            continue
            
    return data_list

# --- CARGAR DATOS ---
parsed_data = parse_input_to_initial_data(RAW_INPUT_DATA)

if parsed_data and len(parsed_data) > 0:
    print(">>> Datos cargados desde RAW_INPUT_DATA")
    initial_data = parsed_data
else:
    print(">>> RAW_INPUT_DATA vacío. Usando valores por defecto.")
    initial_data = [
        {'id': 1, 'x': -0.4, 'y': -4, 'deg': 0.0},
        {'id': 2, 'x':  0.4, 'y': -4, 'deg': 0.0},
        {'id': 3, 'x':  0.0, 'y':  5, 'deg': 180.0}
    ]

# --- GEOMETRÍA ---
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

def calculate_L_metric(trees):
    polys = [t.poly for t in trees]
    union = unary_union(polys)
    minx, miny, maxx, maxy = union.bounds
    w = maxx - minx
    h = maxy - miny
    L = max(h, w/2.0)
    return L, w, h, (minx, miny, maxx, maxy)

def check_collisions_strict(trees):
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].poly.intersects(trees[j].poly):
                # Pequeña tolerancia para bordes que se tocan
                if trees[i].poly.intersection(trees[j].poly).area > 1e-9:
                    return True
    return False

# --- DIBUJADO EN VIVO ---
def update_plot(ax, trees, best_L, bounds, iteration, is_collision):
    ax.clear()
    minx, miny, maxx, maxy = bounds
    
    color = 'red' if is_collision else '#2ca02c'
    
    for t in trees:
        coords = np.array(t.poly.exterior.coords)
        patch = MplPolygon(coords, closed=True, facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(patch)
    
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    box_w = 2 * best_L
    box_h = best_L
    rect = Rectangle((center_x - box_w/2, center_y - box_h/2), 
                     box_w, box_h, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    status = "⚠️ COLISIÓN" if is_collision else "✅ VÁLIDO"
    ax.set_title(f"Iter: {iteration} | {status}\nMejor L: {best_L:.5f}", fontsize=10)
    
    ax.set_aspect('equal')
    ax.set_xlim(minx - 1.0, maxx + 1.0)
    ax.set_ylim(miny - 1.0, maxy + 1.0)
    
    plt.draw()
    plt.pause(PAUSE_TIME)

# --- ALGORITMO PRINCIPAL ---
def optimize_live(start_data, iterations=25000):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    current_trees = [Tree(d['x'], d['y'], d['deg']) for d in start_data]
    
    # Reparación inicial si los datos entrantes ya chocan
    if check_collisions_strict(current_trees):
        print("¡Advertencia! Datos iniciales tienen colisión. Intentando reparar...")
        for _ in range(500):
            for t in current_trees:
                t.x += (np.random.random()-0.5)*0.2
                t.y += (np.random.random()-0.5)*0.2
                t.poly = t.update_poly()
            if not check_collisions_strict(current_trees): 
                print("Reparación exitosa.")
                break
    
    best_trees = copy.deepcopy(current_trees)
    best_L, _, _, bounds = calculate_L_metric(best_trees)
    
    # Parámetros de recocido
    step_pos = 0.005
    step_rot = 0.001
    decay = 0.9995
    
    print(f"Iniciando con L inicial: {best_L:.5f}")
    
    for i in range(iterations):
        candidate_trees = copy.deepcopy(best_trees)
        
        idx = np.random.randint(0, len(candidate_trees))
        candidate_trees[idx].x += (np.random.random() - 0.5) * step_pos
        candidate_trees[idx].y += (np.random.random() - 0.5) * step_pos
        candidate_trees[idx].angle += (np.random.random() - 0.5) * step_rot
        candidate_trees[idx].poly = candidate_trees[idx].update_poly()
        
        collision = check_collisions_strict(candidate_trees)
        
        if not collision:
            new_L, _, _, new_bounds = calculate_L_metric(candidate_trees)
            if new_L < best_L:
                best_L = new_L
                best_trees = candidate_trees
                bounds = new_bounds
                print(f"Iter {i}: Nuevo Récord -> L={best_L:.5f}")
        
        if i % PLOT_EVERY == 0:
            show_trees = candidate_trees 
            update_plot(ax, show_trees, best_L, bounds, i, collision)
        
        step_pos *= decay
        step_rot *= decay
        if step_pos < 0.001: 
            step_pos = 0.02
            step_rot = 1.0

    plt.ioff()
    plt.show()
    return best_trees, best_L

if __name__ == "__main__":
    final_trees, final_L = optimize_live(initial_data)
    
    # OUTPUT FORMATEADO PARA COPIAR Y VOLVER A PEGAR
    print("\n" + "="*40)
    print("COPIA ESTO PARA LA SIGUIENTE RONDA:")
    print("="*40)
    print(f"FINAL L: {final_L}")
    for i, t in enumerate(final_trees):
         print(f"{i+1},s{t.x},s{t.y},s{t.angle},s{final_L}")
    print("="*40)