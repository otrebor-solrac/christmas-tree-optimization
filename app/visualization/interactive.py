import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
import math

# Configurar path para permitir imports de 'app' si se ejecuta como script
if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

try:
    from app.utils.io import load_solution, save_solution
except ImportError:
    load_solution = None
    save_solution = None

# --- 0. CONFIGURACIÓN DE ENTRADA ---
# SI ESTO TIENE DATOS, SE USARÁN. SI ES None, TE PREGUNTARÁ EN CONSOLA.
initial_position = """
FINAL SCORE (SIDE): 1.8692638619865312

"""
# Para introducir número manualmente, cambia lo de arriba por:
# initial_position = None 


# --- CORRECCIÓN DE ATAJOS ---
plt.rcParams['keymap.quit'] = ['ctrl+w', 'cmd+w']
plt.rcParams['keymap.save'] = ['ctrl+s']
plt.rcParams['keymap.pan'] = []
plt.rcParams['keymap.fullscreen'] = []

# --- 1. LÓGICA DE CARGA DE DATOS ---
def parse_raw_input(text):
    """Convierte el texto guardado en lista de diccionarios"""
    if not text or not text.strip():
        return None
        
    data = []
    try:
        lines = text.strip().split('\n')
        for line in lines:
            if "FINAL" in line or not line.strip():
                continue
            parts = line.split(',')
            # Formato esperado: ID, sX, sY, sDeg, sScore
            pid = int(parts[0])
            x = float(parts[1].replace('s', ''))
            y = float(parts[2].replace('s', ''))
            deg = float(parts[3].replace('s', ''))
            data.append({'id': pid, 'x': x, 'y': y, 'deg': deg})
        return data
    except Exception as e:
        print(f"Error parseando input: {e}")
        return None

def setup_initial_trees(input_text):
    print("-" * 40)
    print(" CONFIGURACIÓN DE ÁRBOLES")
    print("-" * 40)
    
    # 1. INTENTAR CARGAR DESDE STRING
    parsed_data = parse_raw_input(input_text)
    
    if parsed_data:
        print(f">>> CARGANDO {len(parsed_data)} ÁRBOLES DESDE 'initial_position'...")
        return parsed_data
    
    # 2. SI NO HAY STRING, PREGUNTAR AL USUARIO
    print(">>> No se detectó posición inicial guardada (es None o inválida).")
    try:
        val = input("¿Cuántos árboles quieres colocar? (Enter para 4): ")
        n = int(val) if val.strip() else 4
    except ValueError:
        n = 4
        
    print(f"\n>>> Generando {n} árboles en cuadrícula...")
    
    data = []
    cols = int(math.ceil(math.sqrt(n)))
    spacing = 1.5
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col - cols/2.0) * spacing + 0.75
        y = (row - cols/2.0) * spacing
        data.append({'id': i + 1, 'x': x, 'y': y, 'deg': 0.0})
    return data

# --- 2. GEOMETRÍA ---
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

# --- 3. CLASES ---
class InteractiveTree:
    def __init__(self, uid, x, y, angle, ax):
        self.id = uid
        self.x = x
        self.y = y
        self.angle = angle
        self.ax = ax
        self.base_poly = ShapelyPolygon(get_tree_coords())
        self.poly = self.update_poly()
        
        coords = np.array(self.poly.exterior.coords)
        self.patch = MplPolygon(coords, closed=True, facecolor='#2ca02c', edgecolor='black', alpha=0.85)
        self.ax.add_patch(self.patch)
        self.text = self.ax.text(self.x, self.y, str(self.id), fontsize=9, color='white', 
                                 ha='center', va='center', fontweight='bold', clip_on=True)

    def update_poly(self):
        r_poly = rotate(self.base_poly, self.angle, origin=(0,0), use_radians=False)
        self.poly = translate(r_poly, xoff=self.x, yoff=self.y)
        return self.poly

    def update_visuals(self, color):
        coords = np.array(self.poly.exterior.coords)
        self.patch.set_xy(coords)
        self.patch.set_facecolor(color)
        self.text.set_position((self.x, self.y))

class Editor:
    def __init__(self, raw_input_data=None, preloaded_data=None):
        # Si hay datos precargados (por argumento), usarlos directamente
        if preloaded_data:
            data = preloaded_data
        else:
            # Pasamos la data cruda a setup_initial_trees
            data = setup_initial_trees(raw_input_data)
            
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.trees = []
        self.dragging_tree = None
        self.hover_tree = None
        self.last_mouse = None
        self.current_score = 0
        
        for d in data:
            t = InteractiveTree(d['id'], d['x'], d['y'], d['deg'], self.ax)
            self.trees.append(t)
            
        self.box_rect = Rectangle((0,0), 1, 1, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
        self.ax.add_patch(self.box_rect)
        
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.grid(True, linestyle=':', alpha=0.4)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('scroll_event', self.on_zoom)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.print_instructions()
        self.update_scene()
        plt.show()

    def print_instructions(self):
        print("\n" + "="*50)
        print(" CONTROLES:")
        print("="*50)
        print(" 🖱️  ARRASTRAR:   Mover Árbol")
        print(" 🖱️  RUEDA:       Zoom In / Zoom Out")
        print(" ⌨️  TECLA 'Q':   Rotar Izquierda")
        print(" ⌨️  TECLA 'E':   Rotar Derecha")
        print(" ⌨️  TECLA 'P':   Guardar solución (CSV)")
        print("="*50 + "\n")

    def check_collisions(self):
        colliding = set()
        is_col = False
        for i in range(len(self.trees)):
            for j in range(i + 1, len(self.trees)):
                if self.trees[i].poly.intersects(self.trees[j].poly):
                    if self.trees[i].poly.intersection(self.trees[j].poly).area > 1e-5:
                        colliding.add(i); colliding.add(j)
                        is_col = True
        return colliding, is_col

    def calculate_metrics(self):
        polys = [t.poly for t in self.trees]
        union = unary_union(polys)
        minx, miny, maxx, maxy = union.bounds
        w, h = maxx - minx, maxy - miny
        side_score = max(h, w)
        return side_score, (minx, miny, maxx, maxy)

    def update_scene(self):
        score, bounds = self.calculate_metrics()
        colliding_indices, is_collision = self.check_collisions()
        
        cx, cy = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
        self.box_rect.set_xy((cx - score/2, cy - score/2))
        self.box_rect.set_width(score)
        self.box_rect.set_height(score)
        
        for i, t in enumerate(self.trees):
            color = '#2ca02c'
            if i in colliding_indices: color = '#d62728'
            if t == self.hover_tree: color = '#1f77b4'
            if t == self.dragging_tree: color = 'orange'
            t.update_visuals(color)
            
        status = "⚠️ CHOQUE" if is_collision else "✅ VÁLIDO"
        self.ax.set_title(f"Métrica Kaggle: {score:.5f} | {status}\nQ/E: Rotar | P: Guardar", fontsize=10)
        self.fig.canvas.draw_idle()
        self.current_score = score

    def on_zoom(self, event):
        if event.inaxes != self.ax: return
        base_scale = 1.2
        scale_factor = 1/base_scale if event.button == 'up' else base_scale
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
        self.ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        self.ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
        self.fig.canvas.draw_idle()

    def on_move(self, event):
        if event.inaxes != self.ax: return
        if self.dragging_tree:
            dx = event.xdata - self.last_mouse[0]
            dy = event.ydata - self.last_mouse[1]
            self.dragging_tree.x += dx
            self.dragging_tree.y += dy
            self.dragging_tree.update_poly()
            self.last_mouse = (event.xdata, event.ydata)
            self.update_scene()
            return
        found = None
        for t in self.trees:
            if t.patch.contains(event)[0]:
                found = t
                break
        if found != self.hover_tree:
            self.hover_tree = found
            self.update_scene()

    def on_click(self, event):
        if event.button != 1 or event.inaxes != self.ax: return
        if self.hover_tree:
            self.dragging_tree = self.hover_tree
            self.last_mouse = (event.xdata, event.ydata)

    def on_release(self, event):
        self.dragging_tree = None
        self.update_scene()

    def on_key(self, event):
        if event.key == 'p':
            # Guardar solución en archivo
            n = len(self.trees)
            filename = f"solutions/T{n}.csv"
            Path("solutions").mkdir(exist_ok=True)
            
            if save_solution:
                # Adaptador para save_solution
                class TempTree:
                    def __init__(self, t):
                        self.id = t.id
                        self.center_x = t.x
                        self.center_y = t.y
                        self.angle = t.angle
                
                adapted_trees = [TempTree(t) for t in self.trees]
                save_solution(filename, adapted_trees, self.current_score)
            else:
                # Fallback manual
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['id', 'x', 'y', 'deg', 'score'])
                    for t in self.trees:
                        writer.writerow([
                            t.id,
                            f"s{t.x}",
                            f"s{t.y}",
                            f"s{t.angle}",
                            f"s{self.current_score}"
                        ])
                print(f"-> Solución guardada en: {filename} (Score: {self.current_score:.4f})")

            print("\n" + "="*40)
            print(f"FINAL SCORE (SIDE): {self.current_score}")
            for t in self.trees:
                print(f"{t.id},s{t.x},s{t.y},s{t.angle},s{self.current_score}")
            print("="*40 + "\n")
            return
        if self.hover_tree and event.key in ['q', 'e']:
            step = 5.0 if event.key == 'q' else -5.0
            self.hover_tree.angle += step
            self.hover_tree.update_poly()
            self.update_scene()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Editor Interactivo de Árboles")
    parser.add_argument("solution_id", nargs="?", type=int, help="ID de la solución a cargar (ej: 24)")
    args = parser.parse_args()

    preloaded_data = None
    
    if args.solution_id is not None:
        if load_solution is None:
            print("Error: No se pudo importar 'load_solution'. Verifica tu entorno.")
            sys.exit(1)
            
        try:
            # Construir ruta a solutions/
            root_dir = Path(__file__).resolve().parent.parent.parent
            csv_path = root_dir / "solutions" / f"T{args.solution_id}.csv"
            
            print(f">>> Cargando solución T{args.solution_id} desde {csv_path}...")
            trees = load_solution(str(csv_path))
            
            # Convertir objetos ChristmasTree a formato de diccionarios del Editor
            preloaded_data = []
            for t in trees:
                preloaded_data.append({
                    'id': t.id,
                    'x': float(t.center_x),
                    'y': float(t.center_y),
                    'deg': float(t.angle)
                })
        except Exception as e:
            print(f"Error cargando solución: {e}")
            sys.exit(1)

    # Si preloaded_data es None, usará initial_position o preguntará al usuario
    app = Editor(raw_input_data=initial_position, preloaded_data=preloaded_data)
