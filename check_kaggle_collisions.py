#!/usr/bin/env python3
"""
Collision checker using EXACT Kaggle metric logic.
Usage: python check_kaggle_collisions.py solutions/T105.csv
"""
from decimal import Decimal, getcontext
from pathlib import Path
import sys

from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

# Decimal precision and scaling factor (EXACT Kaggle values)
getcontext().prec = 25
scale_factor = Decimal('1e18')


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))


def check_kaggle_collisions(csv_path: str) -> list:
    """
    Check for collisions using EXACT Kaggle logic:
    poly.intersects(other) AND NOT poly.touches(other)
    
    Returns list of colliding pairs [(i, j), ...]
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: File {csv_path} not found")
        return []

    # Parse CSV
    trees = []
    with open(path, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                # Remove 's' prefix
                x = parts[1].lstrip('s')
                y = parts[2].lstrip('s')
                deg = parts[3].lstrip('s')
                trees.append(ChristmasTree(x, y, deg))

    print(f"Loaded {len(trees)} trees from {csv_path}")

    # Build R-Tree for efficient spatial queries
    all_polygons = [t.polygon for t in trees]
    r_tree = STRtree(all_polygons)

    # Check for collisions (EXACT Kaggle logic)
    collisions = []
    for i, poly in enumerate(all_polygons):
        indices = r_tree.query(poly)
        for j in indices:
            if j <= i:  # Skip self and already-checked pairs
                continue
            # KAGGLE COLLISION CRITERION:
            # intersects AND NOT touches = TRUE overlap
            if poly.intersects(all_polygons[j]) and not poly.touches(all_polygons[j]):
                collisions.append((i + 1, j + 1))  # Convert to 1-indexed IDs

    return collisions


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_kaggle_collisions.py <solution.csv>")
        print("Example: python check_kaggle_collisions.py solutions/T105.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    collisions = check_kaggle_collisions(csv_path)

    if collisions:
        print(f"\n❌ COLISIONES DETECTADAS: {len(collisions)} pares")
        for i, j in collisions[:20]:  # Show first 20
            print(f"   Árbol {i} <-> Árbol {j}")
        if len(collisions) > 20:
            print(f"   ... y {len(collisions) - 20} más")
    else:
        print("\n✅ NO HAY COLISIONES (Kaggle-compatible)")


if __name__ == "__main__":
    main()
