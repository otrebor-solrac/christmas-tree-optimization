"""
Visualization module for displaying Christmas tree layouts.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.ops import unary_union


class TreeVisualizer:
    """
    Visualizes Christmas tree layouts using matplotlib.
    """
    
    def __init__(self, scale_factor, figsize=(8, 8)):
        self.scale_factor = float(scale_factor)
        self.figsize = figsize
        self.cmap = plt.get_cmap('tab10')

    def _desescalar_coords(self, polygon):
        x, y = polygon.exterior.xy
        x_descaled = [v / self.scale_factor for v in x]
        y_descaled = [v / self.scale_factor for v in y]
        return x_descaled, y_descaled

    def _dibujar_arbol_individual(self, ax, tree, index):
        x, y = self._desescalar_coords(tree.polygon)
        color = self.cmap(index % 10)
        ax.fill(x, y, alpha=0.8, fc=color, ec='black', lw=0.5)

    def _dibujar_bounding_box(self, ax, trees, adjust_limits=True):
        """
        Dibuja la caja y RETORNA EL LADO (SIDE).
        
        Args:
            ax: Eje de matplotlib
            trees: Lista de árboles
            adjust_limits: Si True, ajusta los límites del eje automáticamente. 
                          Si False, solo dibuja el rectángulo sin ajustar límites.
        """
        polys = [t.polygon for t in trees]
        if not polys:
            return 0.0

        u = unary_union(polys)
        minx, miny, maxx, maxy = u.bounds

        w = (maxx - minx) / self.scale_factor
        h = (maxy - miny) / self.scale_factor
        
        # La métrica oficial usa el lado máximo del cuadrado
        side = max(w, h)
        
        origin_x = minx / self.scale_factor
        origin_y = miny / self.scale_factor

        rect = Rectangle(
            (origin_x, origin_y), 
            side, side, 
            lw=2, ec='red', fc='none', ls='--'
        )
        ax.add_patch(rect)
        
        # Solo ajustar límites si se solicita
        if adjust_limits:
            margin = side * 0.1
            margin_x = side * 0.1
            margin_y = side * 0.1
            if margin == 0: 
               margin = 0.5
            
            # Usar autoscale para asegurar que todo se ve primero
            ax.autoscale(True)
            # Luego imponer margen alrededor del bounding box cuadrado para centrarlo
            ax.set_xlim(origin_x - margin, origin_x + side + margin)
            ax.set_ylim(origin_y - margin, origin_y + side + margin)
        
        return side

    def plot(self, trees, score_real=None, check_collisions=None):
        """
        Main method to generate the plot.
        Updates title with Kaggle metric: Area / N
        
        Args:
            trees: Lista de árboles a visualizar
            score_real: Score real a mostrar (opcional)
            check_collisions: Función para verificar colisiones (opcional)
                             Debe retornar (has_collisions: bool, collision_pairs: list)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 1. Dibujar árboles
        for i, tree in enumerate(trees):
            self._dibujar_arbol_individual(ax, tree, i)
        
        # 2. Dibujar caja y obtener el lado visual
        side = self._dibujar_bounding_box(ax, trees)
        
        # 3. CÁLCULO DE LA MÉTRICA KAGGLE PARA N ÁRBOLES
        n = len(trees)
        if n > 0:
            area = side * side
            kaggle_metric_n = area / n  # Esta es la proporción S_n / n
        else:
            area = 0
            kaggle_metric_n = 0
        
        # 4. Verificar colisiones si se proporciona la función
        has_collisions = False
        collision_pairs = []
        collision_message = ""
        
        if check_collisions is not None:
            has_collisions, collision_pairs = check_collisions(trees)
            if has_collisions:
                collision_message = f"\n⚠️  COLISIÓN DETECTADA! Pares: {collision_pairs}"
                print(f"\n{'='*60}")
                print(f"⚠️  COLISIÓN DETECTADA EN LA VISUALIZACIÓN!")
                print(f"   Pares de árboles en colisión: {collision_pairs}")
                print(f"{'='*60}\n")
        
        # 5. Generar Título Informativo
        titulo = f"Visualización - {n} Árboles\n"
        titulo += f"Lado: {side:.4f} | Área: {area:.4f}\n"
        titulo += f"Contribución Kaggle (Area/N): {kaggle_metric_n:.4f}"
        
        if has_collisions:
            titulo += f"\n⚠️ COLISIÓN DETECTADA! Pares: {collision_pairs}"
            # Cambiar color del título a rojo para colisiones
            ax.set_title(titulo, fontsize=10, color='red', weight='bold')
        else:
            ax.set_title(titulo, fontsize=10)
            
        ax.set_aspect('equal')
        plt.grid(True, alpha=0.3, linestyle=':')
        
        print(f"Mostrando gráfico... (Metric: {kaggle_metric_n:.4f})")
        if has_collisions:
            print(f"⚠️  ADVERTENCIA: Hay colisiones en la visualización!")
        plt.show()