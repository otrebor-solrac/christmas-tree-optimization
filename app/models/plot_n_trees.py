import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union

# --- 1. Definición del Árbol ---
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

def create_tree(x, y, angle):
    poly = ShapelyPolygon(get_tree_coords())
    r_poly = rotate(poly, angle, origin=(0,0), use_radians=False)
    t_poly = translate(r_poly, xoff=x, yoff=y)
    return t_poly

def plot_5x5_custom_shift():
    # --- PARÁMETROS BASE ---
    STRIDE_X = 0.412        
    ZIPPER_OFFSET_Y = 0.50 
    ROW_HEIGHT = 0.80     
    
    # --- 🛠️ TU NUEVO PARÁMETRO DE AJUSTE 🛠️ ---
    # El estándar era (STRIDE_X / 2) que es aprox 0.206 (Mueve a la derecha)
    # Para mover a la IZQUIERDA, reduce este número.
    # Prueba: 0.10, 0.0 (alineado), o incluso negativo -0.10
    
    ROW_SHIFT = -0.15  # <--- ¡JUEGA CON ESTE VALOR!
    
    # Configuración de la matriz
    COLS = 20
    ROWS = 10
    N_TOTAL = COLS * ROWS
    GLOBAL_ROTATION = 0.0 

    raw_trees = []
    
    for row in range(ROWS):
        for col in range(COLS):
            x = col * STRIDE_X
            y = row * ROW_HEIGHT
            angle = 0
            
            # Zipper
            if col % 2 != 0:
                angle = 180
                y += ZIPPER_OFFSET_Y
            
            # Lógica de Desplazamiento de Fila (Stagger)
            # Aplicamos el shift solo a las filas impares (1, 3, 5...)
            if row % 2 != 0:
                x += ROW_SHIFT  # Aquí aplicamos tu ajuste manual

            tree = create_tree(x, y, angle)
            raw_trees.append(tree)

    # Rotación Global
    rotated_trees = []
    for t in raw_trees:
        rotated_trees.append(rotate(t, GLOBAL_ROTATION, origin=(0,0), use_radians=False))

    # Bounding Box
    union_poly = unary_union(rotated_trees)
    minx, miny, maxx, maxy = union_poly.bounds
    width = maxx - minx
    height = maxy - miny
    score = (max(width, height)**2) / N_TOTAL

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, tree in enumerate(rotated_trees):
        row_idx = i // COLS
        color = colors[row_idx % len(colors)]
        patch = Polygon(np.array(tree.exterior.coords), closed=True, 
                        facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(patch)
        
    rect = Rectangle((minx, miny), width, height, linewidth=2, 
                     edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    info = (f"Ajuste Manual ROW_SHIFT: {ROW_SHIFT}\n"
            f"Caja: {width:.3f} x {height:.3f}\n"
            f"Score Estimado: {score:.4f}")
    
    ax.set_title(info, fontsize=14)
    print(info)

    margin = 0.5
    ax.set_xlim(minx - margin, maxx + margin)
    ax.set_ylim(miny - margin, maxy + margin)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_5x5_custom_shift()