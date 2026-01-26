"""
Batch script to compress all matching solution files.
"""
import os
import re
import csv
from app.algorithms.compress_shapely import compress_layout
from app.evaluation.cost_evaluator import CostEvaluator
from app.algorithms.compress_shapely import load_trees_from_csv
from app.config import SCALE_FACTOR

SOLUTIONS_DIR = 'solutions'
OUTPUT_DIR = 'solutions_compress'

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    files = [f for f in os.listdir(SOLUTIONS_DIR) if f.endswith('.csv')]
    
    # Filter for T{number}.csv only (no hyphens or other suffixes)
    # Regex: ^T(\d+)\.csv$
    pattern = re.compile(r'^T(\d+)\.csv$')
    
    targets = []
    for f in files:
        match = pattern.match(f)
        if match:
            n = int(match.group(1))
            if n >= 145:  # CONTINUE FROM 67
                targets.append((n, f))
            
    # Sort by N
    targets.sort(key=lambda x: x[0])
    
    print(f"Found {len(targets)} files to compress (starting from N=67).")
    
    results = []
    
    for n, filename in targets:
        input_path = os.path.join(SOLUTIONS_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        print(f"Processing {filename}...")
        try:
            # 1. Evaluate Input
            trees_in = load_trees_from_csv(input_path)
            evaluator = CostEvaluator(SCALE_FACTOR)
            score_in = evaluator.calculate_kaggle_score(trees_in)
            
            # 2. Run Compression to Temp File
            temp_path = output_path + ".tmp"
            compress_layout(input_path, temp_path, iterations=100, step_size=0.1)
            
            # 3. Evaluate Compressed (if saved)
            score_comp = float('inf')
            if os.path.exists(temp_path):
                trees_comp = load_trees_from_csv(temp_path)
                score_comp = evaluator.calculate_kaggle_score(trees_comp)
            
            # 4. DECISION: Input vs Compressed
            import shutil
            
            if score_comp < score_in:
                # Winner: Compressed
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)
                final_score = score_comp
                print(f"  -> IMPROVED! (Original: {score_in:.6f} -> Compressed: {score_comp:.6f})")
                
            else:
                # Winner: Original Input
                shutil.copy2(input_path, output_path)
                final_score = score_in
                print(f"  -> No improvement. Kept original. (Score: {score_in:.6f})")
                if os.path.exists(temp_path): os.remove(temp_path)

            results.append((n, final_score))
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n" + "="*30)
    print("FINAL SCORES (solutions_compress)")
    print("="*30)
    print("N\tScore")
    for n, score in results:
        print(f"{n}\t{score:.6f}")

if __name__ == "__main__":
    main()
