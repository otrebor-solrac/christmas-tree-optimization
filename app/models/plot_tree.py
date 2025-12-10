import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_tree_geometry():
    # --- 1. Definición de Parámetros (Tus datos) ---
    trunk_w, trunk_h = 0.15, 0.2
    base_w, base_y = 0.7, 0.0
    mid_w, tier_2_y = 0.4, 0.25
    top_w, tier_1_y = 0.25, 0.5
    tip_y = 0.8
    trunk_bottom = -trunk_h

    # --- 2. Generación de Coordenadas ---
    # Nota: He ordenado ligeramente los puntos para asegurar que el polígono cierre bien visualmente
    # (Sentido horario desde la punta)
    coords = [
        (0, tip_y),                 # Punta Superior
        (top_w/2, tier_1_y),        # Nivel 1 (Rama Ext)
        (top_w/4, tier_1_y),        # Nivel 1 (Axila)
        (mid_w/2, tier_2_y),        # Nivel 2 (Rama Ext)
        (mid_w/4, tier_2_y),        # Nivel 2 (Axila)
        (base_w/2, base_y),         # Base (Rama Ext)
        (trunk_w/2, base_y),        # Unión Tronco Der
        (trunk_w/2, trunk_bottom),  # Tronco Abajo Der
        (-trunk_w/2, trunk_bottom), # Tronco Abajo Izq
        (-trunk_w/2, base_y),       # Unión Tronco Izq
        (-base_w/2, base_y),        # Base (Rama Izq)
        (-mid_w/4, tier_2_y),       # Nivel 2 (Axila Izq)
        (-mid_w/2, tier_2_y),       # Nivel 2 (Rama Izq)
        (-top_w/4, tier_1_y),       # Nivel 1 (Axila Izq)
        (-top_w/2, tier_1_y)        # Nivel 1 (Rama Izq)
    ]

    # --- 3. Configuración del Gráfico ---
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Crear el polígono
    poly = Polygon(coords, closed=True, facecolor='#2E8B57', edgecolor='black', alpha=0.8, linewidth=2)
    ax.add_patch(poly)

    # --- 4. Marcar y Etiquetar Puntos ---
    # Separar X e Y para scatter plot
    xs, ys = zip(*coords)
    ax.scatter(xs, ys, color='red', zorder=5, s=50)

    # Añadir texto a cada punto
    for x, y in coords:
        # Lógica simple para desplazar el texto y que no tape la línea
        offset_x = 0.02 if x >= 0 else -0.12
        offset_y = 0.01
        
        # Formato de etiqueta: (0.12, 0.50)
        label = f"({x:.3f}, {y:.2f})"
        
        ax.annotate(label, 
                    (x, y), 
                    xytext=(x + offset_x, y + offset_y),
                    fontsize=9,
                    weight='bold',
                    color='darkblue')

    # --- 5. Dibujar Niveles (Líneas Guía) ---
    # Líneas horizontales para ver dónde están los cortes (Axilas)
    niveles = [0.0, 0.25, 0.50]
    for n in niveles:
        ax.axhline(n, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(0.4, n + 0.01, f"Y={n}", color='gray', fontsize=8)

    # --- 6. Ajustes Finales ---
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.3, 0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Geometría Exacta del Árbol\nÁrea Total Aprox: {0.2456}", fontsize=14)
    ax.set_xlabel("Ancho (X)")
    ax.set_ylabel("Alto (Y)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_tree_geometry()