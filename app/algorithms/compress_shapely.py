"""
Script to compress a Christmas tree layout using Python's shapely library for collision detection.
This script aims to squeeze trees closer to the origin to improve the Kaggle score.
"""

import sys
import os
import argparse
import csv
import math
from decimal import Decimal, getcontext

# Add parent directory to sys.path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.models.tree import ChristmasTree
from app.evaluation.cost_evaluator import CostEvaluator
from app.config import SCALE_FACTOR

# Set decimal precision
getcontext().prec = 28

def load_trees_from_csv(filepath):
    """Loads trees from a CSV file."""
    trees = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle potential 's' prefix in scientific notation if present (from Rust output)
            x = row['x'].replace('s', '') if 's' in row['x'] else row['x']
            y = row['y'].replace('s', '') if 's' in row['y'] else row['y']
            deg = row['deg'].replace('s', '') if 's' in row['deg'] else row['deg']
            
            trees.append(ChristmasTree(
                int(row['id']),
                x,
                y,
                deg
            ))
    return trees

def save_trees_to_csv(trees, filepath, score):
    """Saves trees to a CSV file."""
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'deg', 'score'])
        for t in trees:
            writer.writerow([
                t.id,
                f"s{t.center_x}",
                f"s{t.center_y}",
                f"s{t.angle}",
                f"s{score}"
            ])
    print(f"Saved optimized solution to {filepath} with score {score}")

def compress_layout(input_file, output_file, iterations=1000, step_size=0.1):
    """
    Compresses the tree layout by pulling trees towards the center.
    """
    print(f"Loading from {input_file}...")
    trees = load_trees_from_csv(input_file)
    
    evaluator = CostEvaluator(SCALE_FACTOR)
    current_score = evaluator.calculate_kaggle_score(trees)
    initial_score = current_score
    
    print(f"Initial Score: {initial_score}")
    
    # Check initial collision
    overlap = evaluator.calculate_soft_overlap(trees)
    if overlap > 1e-9:
        print(f"WARNING: Initial layout has overlap: {overlap}")
    
    improved = False
    
    for i in range(iterations):
        # Decay step size
        current_step = step_size * (1 - (i / iterations))
        if current_step < 1e-6:
            current_step = 1e-6
            
        tree_moved = False
        
        # Sort trees by distance to center (furthest first might open space?)
        # Or closest first to pack center tight? Let's try iterating all.
        # Actually random order or sorted by ID is fine for now.
        
        for tree in trees:
            # Calculate vector to origin
            x = float(tree.center_x)
            y = float(tree.center_y)
            dist = math.sqrt(x*x + y*y)
            
            if dist < 1e-6:
                continue
                
            # Direction to center
            dx = -x / dist * current_step
            dy = -y / dist * current_step
            
            # Save original position
            orig_x = tree.center_x
            orig_y = tree.center_y
            
            # Apply move
            tree.center_x += Decimal(dx)
            tree.center_y += Decimal(dy)
            tree.update_polygon()
            
            # Validate
            # 1. Check overlap
            if evaluator.calculate_soft_overlap(trees) > 1e-9:
                # Revert if overlap
                tree.center_x = orig_x
                tree.center_y = orig_y
                tree.update_polygon()
                continue

            # 2. STRICT SCORE CHECK
            # We only accept moves that maintain or improve the Kaggle Score.
            # In SA we might accept worse, but here we are doing "Greedy Compression".
            new_score = evaluator.calculate_kaggle_score(trees)
            if new_score > current_score + 1e-12: # Allow tiny floating point noise
                # Revert if score worsened
                tree.center_x = orig_x
                tree.center_y = orig_y
                tree.update_polygon()
            else:
                # Accepted
                current_score = new_score
                tree_moved = True
                improved = True
                
        if (i + 1) % 100 == 0:
            print(f"Iter {i+1}: Score {current_score:.6f}, Step {current_step:.6f}")

    final_score = evaluator.calculate_kaggle_score(trees)
    print(f"Final Score: {final_score}")
    
    # Sólo guardamos si es estrictamente mejor (o igual, para actualizar posiciones)
    # Pero el usuario se queja de score más alto, así que seamos estrictos.
    if final_score < initial_score - 1e-9:
        save_trees_to_csv(trees, output_file, final_score)
    else:
        print(f"No meaningful improvement. (Initial: {initial_score}, Final: {final_score})")
        # Optional: Save anyway if it didn't get worse? No, user wants better.
        if not os.path.exists(output_file) and final_score <= initial_score:
             # Si no existe el archivo de salida, guardamos aunque sea igual
             save_trees_to_csv(trees, output_file, final_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compress Christmas tree layout.')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('output', help='Output CSV file')
    parser.add_argument('--iters', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--step', type=float, default=0.1, help='Initial step size')
    
    args = parser.parse_args()
    
    compress_layout(args.input, args.output, args.iters, args.step)
