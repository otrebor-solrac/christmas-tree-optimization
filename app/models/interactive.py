import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
import math

# --- 0. CONFIGURACIÓN DE ENTRADA ---
# SI ESTO TIENE DATOS, SE USARÁN. SI ES None, TE PREGUNTARÁ EN CONSOLA.
initial_position = """
FINAL SCORE (SIDE): 1.8692638619865312
1,s-3.232888888888889,s-3.448366013071897,s0.0,s0.4070904238957436
2,s-2.820388888888889,s-2.948366013071897,s180.0,s0.4070904238957436
3,s-2.4078888888888894,s-3.448366013071897,s0.0,s0.4070904238957436
4,s-1.9953888888888893,s-2.948366013071897,s180.0,s0.4070904238957436
5,s-1.5828888888888892,s-3.448366013071897,s0.0,s0.4070904238957436
6,s-1.1703888888888891,s-2.948366013071897,s180.0,s0.4070904238957436
7,s-0.7578888888888895,s-3.448366013071897,s0.0,s0.4070904238957436
8,s-0.3453888888888894,s-2.948366013071897,s180.0,s0.4070904238957436
9,s0.06711111111111068,s-3.448366013071897,s0.0,s0.4070904238957436
10,s0.47961111111111077,s-2.948366013071897,s180.0,s0.4070904238957436
11,s0.8921111111111109,s-3.448366013071897,s0.0,s0.4070904238957436
12,s1.3046111111111105,s-2.948366013071897,s180.0,s0.4070904238957436
13,s1.7171111111111101,s-3.448366013071897,s0.0,s0.4070904238957436
14,s2.1296111111111107,s-2.948366013071897,s180.0,s0.4070904238957436
15,s2.5421111111111103,s-3.448366013071897,s0.0,s0.4070904238957436
16,s2.954611111111111,s-2.948366013071897,s180.0,s0.4070904238957436
17,s3.3671111111111105,s-3.448366013071897,s0.0,s0.4070904238957436
18,s-3.383888888888889,s-2.1483660130718967,s180.0,s0.4070904238957436
19,s-2.9713888888888893,s-2.6483660130718967,s0.0,s0.4070904238957436
20,s-2.558888888888889,s-2.1483660130718967,s180.0,s0.4070904238957436
21,s-2.1463888888888896,s-2.6483660130718967,s0.0,s0.4070904238957436
22,s-1.7338888888888893,s-2.1483660130718967,s180.0,s0.4070904238957436
23,s-1.3213888888888892,s-2.6483660130718967,s0.0,s0.4070904238957436
24,s-0.9088888888888893,s-2.1483660130718967,s180.0,s0.4070904238957436
25,s-0.4963888888888892,s-2.6483660130718967,s0.0,s0.4070904238957436
26,s-0.08388888888888912,s-2.1483660130718967,s180.0,s0.4070904238957436
27,s0.32861111111111097,s-2.6483660130718967,s0.0,s0.4070904238957436
28,s0.741111111111111,s-2.1483660130718967,s180.0,s0.4070904238957436
29,s1.1536111111111107,s-2.6483660130718967,s0.0,s0.4070904238957436
30,s1.5661111111111103,s-2.1483660130718967,s180.0,s0.4070904238957436
31,s1.9786111111111109,s-2.6483660130718967,s0.0,s0.4070904238957436
32,s2.3911111111111105,s-2.1483660130718967,s180.0,s0.4070904238957436
33,s2.803611111111111,s-2.6483660130718967,s0.0,s0.4070904238957436
34,s3.2161111111111107,s-2.1483660130718967,s180.0,s0.4070904238957436
35,s-3.232888888888889,s-1.8483660130718969,s0.0,s0.4070904238957436
36,s-2.820388888888889,s-1.3483660130718969,s180.0,s0.4070904238957436
37,s-2.4078888888888894,s-1.8483660130718969,s0.0,s0.4070904238957436
38,s-1.9953888888888893,s-1.3483660130718969,s180.0,s0.4070904238957436
39,s-1.5828888888888892,s-1.8483660130718969,s0.0,s0.4070904238957436
40,s-1.1703888888888891,s-1.3483660130718969,s180.0,s0.4070904238957436
41,s-0.7578888888888895,s-1.8483660130718969,s0.0,s0.4070904238957436
42,s-0.3453888888888894,s-1.3483660130718969,s180.0,s0.4070904238957436
43,s0.06711111111111068,s-1.8483660130718969,s0.0,s0.4070904238957436
44,s0.47961111111111077,s-1.3483660130718969,s180.0,s0.4070904238957436
45,s0.8921111111111109,s-1.8483660130718969,s0.0,s0.4070904238957436
46,s1.3046111111111105,s-1.3483660130718969,s180.0,s0.4070904238957436
47,s1.7171111111111101,s-1.8483660130718969,s0.0,s0.4070904238957436
48,s2.1296111111111107,s-1.3483660130718969,s180.0,s0.4070904238957436
49,s2.5421111111111103,s-1.8483660130718969,s0.0,s0.4070904238957436
50,s2.954611111111111,s-1.3483660130718969,s180.0,s0.4070904238957436
51,s3.3671111111111105,s-1.8483660130718969,s0.0,s0.4070904238957436
52,s-3.383888888888889,s-0.5483660130718966,s180.0,s0.4070904238957436
53,s-2.9713888888888893,s-1.0483660130718966,s0.0,s0.4070904238957436
54,s-2.558888888888889,s-0.5483660130718966,s180.0,s0.4070904238957436
55,s-2.1463888888888896,s-1.0483660130718966,s0.0,s0.4070904238957436
56,s-1.7338888888888893,s-0.5483660130718966,s180.0,s0.4070904238957436
57,s-1.3213888888888892,s-1.0483660130718966,s0.0,s0.4070904238957436
58,s-0.9088888888888893,s-0.5483660130718966,s180.0,s0.4070904238957436
59,s-0.4963888888888892,s-1.0483660130718966,s0.0,s0.4070904238957436
60,s-0.08388888888888912,s-0.5483660130718966,s180.0,s0.4070904238957436
61,s0.32861111111111097,s-1.0483660130718966,s0.0,s0.4070904238957436
62,s0.741111111111111,s-0.5483660130718966,s180.0,s0.4070904238957436
63,s1.1536111111111107,s-1.0483660130718966,s0.0,s0.4070904238957436
64,s1.5661111111111103,s-0.5483660130718966,s180.0,s0.4070904238957436
65,s1.9786111111111109,s-1.0483660130718966,s0.0,s0.4070904238957436
66,s2.3911111111111105,s-0.5483660130718966,s180.0,s0.4070904238957436
67,s2.803611111111111,s-1.0483660130718966,s0.0,s0.4070904238957436
68,s3.2161111111111107,s-0.5483660130718966,s180.0,s0.4070904238957436
69,s-3.232888888888889,s-0.24836601307189676,s0.0,s0.4070904238957436
70,s-2.820388888888889,s0.25163398692810324,s180.0,s0.4070904238957436
71,s-2.4078888888888894,s-0.24836601307189676,s0.0,s0.4070904238957436
72,s-1.9953888888888893,s0.25163398692810324,s180.0,s0.4070904238957436
73,s-1.5828888888888892,s-0.24836601307189676,s0.0,s0.4070904238957436
74,s-1.1703888888888891,s0.25163398692810324,s180.0,s0.4070904238957436
75,s-0.7578888888888895,s-0.24836601307189676,s0.0,s0.4070904238957436
76,s-0.3453888888888894,s0.25163398692810324,s180.0,s0.4070904238957436
77,s0.06711111111111068,s-0.24836601307189676,s0.0,s0.4070904238957436
78,s0.47961111111111077,s0.25163398692810324,s180.0,s0.4070904238957436
79,s0.8921111111111109,s-0.24836601307189676,s0.0,s0.4070904238957436
80,s1.3046111111111105,s0.25163398692810324,s180.0,s0.4070904238957436
81,s1.7171111111111101,s-0.24836601307189676,s0.0,s0.4070904238957436
82,s2.1296111111111107,s0.25163398692810324,s180.0,s0.4070904238957436
83,s2.5421111111111103,s-0.24836601307189676,s0.0,s0.4070904238957436
84,s2.954611111111111,s0.25163398692810324,s180.0,s0.4070904238957436
85,s3.3671111111111105,s-0.24836601307189676,s0.0,s0.4070904238957436
86,s-3.383888888888889,s1.051633986928103,s180.0,s0.4070904238957436
87,s-2.9713888888888893,s0.5516339869281031,s0.0,s0.4070904238957436
88,s-2.558888888888889,s1.051633986928103,s180.0,s0.4070904238957436
89,s-2.1463888888888896,s0.5516339869281031,s0.0,s0.4070904238957436
90,s-1.7338888888888893,s1.051633986928103,s180.0,s0.4070904238957436
91,s-1.3213888888888892,s0.5516339869281031,s0.0,s0.4070904238957436
92,s-0.9088888888888893,s1.051633986928103,s180.0,s0.4070904238957436
93,s-0.4963888888888892,s0.5516339869281031,s0.0,s0.4070904238957436
94,s-0.08388888888888912,s1.051633986928103,s180.0,s0.4070904238957436
95,s0.32861111111111097,s0.5516339869281031,s0.0,s0.4070904238957436
96,s0.741111111111111,s1.051633986928103,s180.0,s0.4070904238957436
97,s1.1536111111111107,s0.5516339869281031,s0.0,s0.4070904238957436
98,s1.5661111111111103,s1.051633986928103,s180.0,s0.4070904238957436
99,s1.9786111111111109,s0.5516339869281031,s0.0,s0.4070904238957436
100,s2.3911111111111105,s1.051633986928103,s180.0,s0.4070904238957436
101,s2.803611111111111,s0.5516339869281031,s0.0,s0.4070904238957436
102,s3.2161111111111107,s1.051633986928103,s180.0,s0.4070904238957436
103,s-3.232888888888889,s1.3516339869281038,s0.0,s0.4070904238957436
104,s-2.820388888888889,s1.8516339869281038,s180.0,s0.4070904238957436
105,s-2.4078888888888894,s1.3516339869281038,s0.0,s0.4070904238957436
106,s-1.9953888888888893,s1.8516339869281038,s180.0,s0.4070904238957436
107,s-1.5828888888888892,s1.3516339869281038,s0.0,s0.4070904238957436
108,s-1.1703888888888891,s1.8516339869281038,s180.0,s0.4070904238957436
109,s-0.7578888888888895,s1.3516339869281038,s0.0,s0.4070904238957436
110,s-0.3453888888888894,s1.8516339869281038,s180.0,s0.4070904238957436
111,s0.06711111111111068,s1.3516339869281038,s0.0,s0.4070904238957436
112,s0.47961111111111077,s1.8516339869281038,s180.0,s0.4070904238957436
113,s0.8921111111111109,s1.3516339869281038,s0.0,s0.4070904238957436
114,s1.3046111111111105,s1.8516339869281038,s180.0,s0.4070904238957436
115,s1.7171111111111101,s1.3516339869281038,s0.0,s0.4070904238957436
116,s2.1296111111111107,s1.8516339869281038,s180.0,s0.4070904238957436
117,s2.5421111111111103,s1.3516339869281038,s0.0,s0.4070904238957436
118,s2.954611111111111,s1.8516339869281038,s180.0,s0.4070904238957436
119,s3.3671111111111105,s1.3516339869281038,s0.0,s0.4070904238957436
120,s-3.383888888888889,s2.6516339869281036,s180.0,s0.4070904238957436
121,s-2.9713888888888893,s2.1516339869281036,s0.0,s0.4070904238957436
122,s-2.558888888888889,s2.6516339869281036,s180.0,s0.4070904238957436
123,s-2.1463888888888896,s2.1516339869281036,s0.0,s0.4070904238957436
124,s-1.7338888888888893,s2.6516339869281036,s180.0,s0.4070904238957436
125,s-1.3213888888888892,s2.1516339869281036,s0.0,s0.4070904238957436
126,s-0.9088888888888893,s2.6516339869281036,s180.0,s0.4070904238957436
127,s-0.4963888888888892,s2.1516339869281036,s0.0,s0.4070904238957436
128,s-0.08388888888888912,s2.6516339869281036,s180.0,s0.4070904238957436
129,s0.32861111111111097,s2.1516339869281036,s0.0,s0.4070904238957436
130,s0.741111111111111,s2.6516339869281036,s180.0,s0.4070904238957436
131,s1.1536111111111107,s2.1516339869281036,s0.0,s0.4070904238957436
132,s1.5661111111111103,s2.6516339869281036,s180.0,s0.4070904238957436
133,s1.9786111111111109,s2.1516339869281036,s0.0,s0.4070904238957436
134,s2.3911111111111105,s2.6516339869281036,s180.0,s0.4070904238957436
135,s2.803611111111111,s2.1516339869281036,s0.0,s0.4070904238957436
136,s3.2161111111111107,s2.6516339869281036,s180.0,s0.4070904238957436
137,s-3.232888888888889,s2.9516339869281034,s0.0,s0.4070904238957436
138,s-2.820388888888889,s3.4516339869281034,s180.0,s0.4070904238957436
139,s-2.4078888888888894,s2.9516339869281034,s0.0,s0.4070904238957436
140,s-1.9953888888888893,s3.4516339869281034,s180.0,s0.4070904238957436
141,s-1.5828888888888892,s2.9516339869281034,s0.0,s0.4070904238957436
142,s-1.1703888888888891,s3.4516339869281034,s180.0,s0.4070904238957436
143,s-0.7578888888888895,s2.9516339869281034,s0.0,s0.4070904238957436
144,s-0.3453888888888894,s3.4516339869281034,s180.0,s0.4070904238957436
145,s0.06711111111111068,s2.9516339869281034,s0.0,s0.4070904238957436
146,s0.47961111111111077,s3.4516339869281034,s180.0,s0.4070904238957436
147,s0.8921111111111109,s2.9516339869281034,s0.0,s0.4070904238957436
148,s1.3046111111111105,s3.4516339869281034,s180.0,s0.4070904238957436
149,s1.7171111111111101,s2.9516339869281034,s0.0,s0.4070904238957436
150,s2.1296111111111107,s3.4516339869281034,s180.0,s0.4070904238957436
151,s2.5421111111111103,s2.9516339869281034,s0.0,s0.4070904238957436
152,s2.954611111111111,s3.4516339869281034,s180.0,s0.4070904238957436
153,s3.3671111111111105,s2.9516339869281034,s0.0,s0.4070904238957436
154,s3.63391027586054,s2.1516339869281036,s0,s0.4070904238957436
155,s5.63391027586054,s2.1516339869281036,s0,s0.4070904238957436
156,s5.63391027586054,s2.1516339869281036,s0,s0.4070904238957436
"""
# Para introducir número manualmente, cambia lo de arriba por:
initial_position = None 


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
    def __init__(self, raw_input_data):
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
        print(" ⌨️  TECLA 'P':   Imprimir datos")
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
    # Pasamos la variable initial_position al Editor
    app = Editor(initial_position)