"""
Input/Output utilities for the Christmas Tree Packing application.
Handles reading and writing of tree configurations to CSV files.
"""
import csv
from pathlib import Path
from decimal import Decimal
from ..models.tree import ChristmasTree

def save_solution(filename, trees, score):
    """
    Saves the list of trees to a CSV file.
    
    Format: id,x,y,deg,score
    Values are prefixed with 's' to preserve string precision.
    
    Args:
        filename (str): Path to the output CSV file.
        trees (list): List of ChristmasTree objects.
        score (float): The Kaggle score (Area/N) of this solution.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'deg', 'score'])
        
        for tree in trees:
            writer.writerow([
                tree.id,
                f"s{tree.center_x}",
                f"s{tree.center_y}",
                f"s{tree.angle}",
                f"s{score}"
            ])
    print(f"-> Solución guardada en: {filename} (Score: {score:.4f})")

def load_solution(filename):
    """
    Loads trees from a CSV file.
    
    Args:
        filename (str): Path to the input CSV file.
        
    Returns:
        list: List of ChristmasTree objects.
    """
    trees = []
    path = Path(filename)
    
    if not path.exists():
        raise FileNotFoundError(f"El archivo {filename} no existe.")
        
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip 's' prefix and convert to Decimal
            x = row['x'].lstrip('s')
            y = row['y'].lstrip('s')
            deg = row['deg'].lstrip('s')
            
            tree = ChristmasTree(
                id_tree=int(row['id']),
                center_x=x,
                center_y=y,
                angle=deg,
                fixed=False # Load as mutable by default to allow further optimization
            )
            trees.append(tree)
            
    print(f"-> Cargados {len(trees)} árboles desde {filename}")
    return trees
