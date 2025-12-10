import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import translate, rotate

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

def plot_tight_rotated_zipper():
    # --- PARÁMETROS EXTREMOS ---
    # 1. Configuración de "Un nivel arriba" (Deep Zipper)
    STRIDE_X = 0.46
    ZIPPER_OFFSET_Y = 0.25  # Intentamos meterlo hasta la axila inferior
    
    # 2. Rotación Geométrica
    # Usamos 45.0 grados porque coincide con la pendiente de la base del árbol
    GLOBAL_ROTATION = 26.2

    # --- Generación ---
    # Paso A: Crear pareja vertical (que sabemos que se solapa un poco)
    t1_base = create_tree(0, 0, 0)
    t2_base = create_tree(STRIDE_X, ZIPPER_OFFSET_Y, 180)

    # Paso B: Rotar el conjunto COMPLETO 45 grados
    # Rotamos alrededor del (0,0) para mantener la relación relativa
    t1_rot = rotate(t1_base, GLOBAL_ROTATION, origin=(0,0), use_radians=False)
    t2_rot = rotate(t2_base, GLOBAL_ROTATION, origin=(0,0), use_radians=False)
    
    trees = [t1_rot, t2_rot]
    colors = ['#1f77b4', '#ff7f0e']

    # --- Calcular Bounding Box ---
    union_poly = t1_rot.union(t2_rot)
    minx, miny, maxx, maxy = union_poly.bounds
    width = maxx - minx
    height = maxy - miny
    
    # Score Potencial (Ignorando colisión por un momento para ver la geometría)
    lado_max = max(width, height)
    score = (lado_max**2) / 2

    # --- GRAFICAR ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Dibujar árboles
    for i, tree in enumerate(trees):
        patch = Polygon(np.array(tree.exterior.coords), closed=True, 
                        facecolor=colors[i], edgecolor='black', alpha=0.7)
        ax.add_patch(patch)
        
        c = tree.centroid
        ax.text(c.x, c.y, f"T{i+1}", ha='center', va='center', 
                color='white', fontweight='bold', rotation=GLOBAL_ROTATION)

    # Dibujar Caja Roja
    rect = Rectangle((minx, miny), width, height, linewidth=2, 
                     edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # --- ANÁLISIS DE ALINEACIÓN ---
    # Dibujamos líneas guía para ver si los bordes del árbol se alinean con la caja
    # La base del árbol original es horizontal. Al rotar 45, debería quedar diagonal...
    # PERO la pendiente lateral de la base del árbol es 45 grados.
    # Al rotar 45 grados, esa pendiente debería volverse VERTICAL u HORIZONTAL.
    ax.axhline(miny, color='green', linestyle=':', alpha=0.5, label="Límites Caja")
    ax.axvline(minx, color='green', linestyle=':', alpha=0.5)

    # Verificar Colisión
    if t1_rot.intersects(t2_rot):
        collision_area = t1_rot.intersection(t2_rot).area
        # ax.text(0.5, 1.02, f"⚠️ SOLAPAMIENTO: {collision_area:.4f} ⚠️", 
        #         transform=ax.transAxes, ha='center', color='darkred', fontweight='bold')
        print(f"⚠️ SOLAPAMIENTO: {collision_area:.4f} ⚠️")
    info = (f"Cremallera Compacta (OffY=0.25) + Rotación {GLOBAL_ROTATION}°\n"
            f"Box: {width:.3f} x {height:.3f}\n"
            f"Score Potencial: {score:.4f}")
    print(info)
    # ax.set_title(info, fontsize=13)
    
    # Ajustes
    margin = 0.4
    ax.set_xlim(minx - margin, maxx + margin)
    ax.set_ylim(miny - margin, maxy + margin)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_tight_rotated_zipper()