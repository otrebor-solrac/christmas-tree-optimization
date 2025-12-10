"""
Cost evaluation module for the packing algorithm.
Calculates scores and overlap metrics for tree layouts.
"""
from shapely.ops import unary_union


class CostEvaluator:
    """
    Evaluates the cost/score of a tree layout.
    
    Provides methods to calculate:
    - Bounds score: The size of the bounding box containing all trees
    - Soft overlap: The amount of overlapping area between trees
    """
    
    def __init__(self, scale_factor):
        """
        Initialize the cost evaluator.
        
        Args:
            scale_factor: The scale factor used in calculations (Decimal or float)
        """
        self.scale_factor = float(scale_factor)
        self.sf_sq = self.scale_factor ** 2

    def calculate_bounds_score(self, trees):
        """
        Calculate the bounding box score (the side length of the square containing all trees).
        
        Args:
            trees: List of ChristmasTree objects
            
        Returns:
            float: The maximum dimension (width or height) of the bounding box
        """
        polys = [t.polygon for t in trees]
        if not polys:
            return 0.0
        try:
            union_poly = unary_union(polys)
            minx, miny, maxx, maxy = union_poly.bounds
            w = maxx - minx
            h = maxy - miny
            return max(w, h) / self.scale_factor
        
        except Exception:
            return 1000.0

    def calculate_soft_overlap(self, trees) -> float:
        """
        Calculate the soft overlap metric (total overlapping area between trees).
        
        Args:
            trees: List of ChristmasTree objects
            
        Returns:
            float: The normalized overlap area
        """
        polys = [t.polygon for t in trees]
        if not polys:
            return 0.0
        total_area = sum(p.area for p in polys)
        try:
            u_area = unary_union(polys).area
            overlap = total_area - u_area
        except Exception:
            overlap = 1.0 * self.sf_sq
        return overlap / self.sf_sq

    def calculate_kaggle_score(self, trees):
            """
            CALCULA EL SCORE OFICIAL DE KAGGLE.

            Fórmula: Score = (Lado * Lado) / N
            """
            if not trees:
                return 0.0
            side = self.calculate_bounds_score(trees)
            area = side * side
            return area / len(trees)


    def get_total_energy(self, trees, penalty_weight):
        """
        Calcula la ENERGÍA TOTAL para el algoritmo de Simulated Annealing.
        
        Formula: 
        Energy = (MaxSide^2) + (AspectPenalty * 2) + (Overlap * PenaltyWeight)
        
        - MaxSide^2: Buscamos minimizar el área del cuadrado (objetivo Kaggle).
        - AspectPenalty: Castiga si Ancho != Alto para evitar líneas o torres.
        - Overlap: Castigo dinámico para evitar colisiones.
        """
        # 1. Calcular Overlap
        overlap = self.calculate_soft_overlap(trees)
        
        # 2. Calcular Dimensiones y Geometría
        polys = [t.polygon for t in trees]
        if not polys: 
            return 0.0
            
        try:
            union_poly = unary_union(polys)
            minx, miny, maxx, maxy = union_poly.bounds
            
            # Dimensiones desescaladas
            width = (maxx - minx) / self.scale_factor
            height = (maxy - miny) / self.scale_factor
            
            # Score Base: El objetivo principal es minimizar el cuadrado contenedor
            base_score = max(width, height) ** 2
            
            # Penalización por Aspect Ratio (Rectangularidad)
            # Si width y height son muy diferentes, sumamos costo extra.
            # Esto ayuda a corregir la "Deriva Diagonal" o las torres.
            aspect_penalty = abs(width - height) * 2.0 
            
            # Energía Total Combinada
            return base_score + aspect_penalty + (overlap * penalty_weight)
            
        except Exception:
            return 1000.0

