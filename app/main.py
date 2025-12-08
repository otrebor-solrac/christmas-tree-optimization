"""
Main entry point for the Christmas Tree Packing application.
"""
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import SCALE_FACTOR
from app.visualization import TreeVisualizer
from app.algorithms import PackingAlgorithm


def main():
    """Main execution function."""
    # Initialize components
    viz = TreeVisualizer(SCALE_FACTOR)
    solver = PackingAlgorithm(SCALE_FACTOR)

    # Add initial fixed trees
    # solver.add_configured_tree(1, 0.0, 0.0, 45, fixed=False)
    # solver.add_configured_tree(2, -0.05, 0.75, 225, fixed=False)
    # solver.add_configured_tree(3, 5, 0.20, 135, fixed=True)

    final_trees = solver.add_trees(n_new=2, visualizer=viz)

    # Verificar colisiones antes de visualizar
    viz.plot(final_trees, score_real=solver.best_score, check_collisions=solver.detect_collisions)


if __name__ == "__main__":
    main()

