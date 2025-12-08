"""
ChristmasTree model class.
Represents a single Christmas tree with its geometry and position.
"""
from decimal import Decimal
from shapely import affinity
from shapely.geometry import Polygon

from ..config import SCALE_FACTOR


class ChristmasTree:
    """
    Represents a Christmas tree with its position, rotation, and geometry.
    
    Attributes:
        id: Unique identifier for the tree
        center_x: X coordinate of the tree center (Decimal)
        center_y: Y coordinate of the tree center (Decimal)
        angle: Rotation angle in degrees (Decimal)
        fixed: Whether the tree position is fixed (cannot be moved)
        poly_base: Base polygon geometry (before rotation and translation)
        polygon: Final polygon geometry (after rotation and translation)
    """
    
    def __init__(self, id_tree, center_x='0', center_y='0', angle='0', fixed=False):
        """
        Initialize a Christmas tree.
        
        Args:
            id_tree: Unique identifier for the tree
            center_x: X coordinate of center (string or number)
            center_y: Y coordinate of center (string or number)
            angle: Rotation angle in degrees (string or number)
            fixed: Whether the tree position is fixed
        """
        self.id = id_tree
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        self.fixed = fixed
        
        # Geometría base escalada
        sf = float(SCALE_FACTOR)
        # Definición simplificada para Shapely (pero manteniendo escala)
        # Usamos tus medidas:
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
        # Escalar coordenadas
        scaled_coords = [(x * sf, y * sf) for x, y in coords]
        self.poly_base = Polygon(scaled_coords)
        self.update_polygon()

    def update_polygon(self):
        """
        Update the polygon geometry based on current position and rotation.
        This method should be called whenever center_x, center_y, or angle changes.
        """
        # Rotar y trasladar
        sf = float(SCALE_FACTOR)
        rotated = affinity.rotate(self.poly_base, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x) * sf,
            yoff=float(self.center_y) * sf
        )

