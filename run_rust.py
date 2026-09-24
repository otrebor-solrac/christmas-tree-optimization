#!/usr/bin/env python3
"""
Compile and run the Rust Christmas Tree Optimizer.
"""
import subprocess
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Compile and run the Rust Christmas Tree Optimizer")
    parser.add_argument("--target", type=int, required=True, help="Number of trees (N) to optimize (e.g. 20)")
    parser.add_argument("--pop", type=int, default=50, help="Population size (default: 50)")
    parser.add_argument("--gen", type=int, default=100, help="Number of generations (default: 100)")
    parser.add_argument("--steps", type=int, default=50, help="Gravity steps (default: 50)")
    parser.add_argument("--fine-tune", type=int, default=0, help="Fine-tuning iterations at the end (default: 0)")
    parser.add_argument("--groups", action="store_true", help="Enable group mode (pairs of trees coupled together for GA)")
    parser.add_argument("--plot", action="store_true", help="Automatically generate and display/save graph after optimization")
    parser.add_argument("--strategy", type=str, default="sa", help="Strategy: sa, ga, finetune, gravity, deca, zipskew, all")
    parser.add_argument("--optimizer", type=str, default="none", help="Alias for strategy (e.g. sa, ga, cmaes)")
    
    args, extra_args = parser.parse_known_args()
    
    # Paths
    root_dir = Path(__file__).parent.resolve()
    rust_dir = root_dir / "rust_optimizer"
    solutions_dir = root_dir / "solutions"
    binary_path = rust_dir / "target" / "release" / "christmas_tree_optimizer"
    
    solutions_dir.mkdir(parents=True, exist_ok=True)
    output_file = solutions_dir / f"T{args.target}.csv"
    
    # 1. Compile Rust (Release mode)
    print("🔨 Compiling Rust project...")
    try:
        subprocess.run(["cargo", "build", "--release"], cwd=rust_dir, check=True)
    except Exception as e:
        print(f"❌ Rust compilation failed: {e}")
        sys.exit(1)

    # Resolve strategy
    strategy = args.optimizer if args.optimizer != "none" else args.strategy

    # 2. Execute Binary
    print(f"\n🚀 Launching Rust Pipeline for T{args.target} with strategy '{strategy}'...")
    
    cmd = [
        str(binary_path),
        "--input", str(output_file),
        "--output", str(output_file),
        "--target-n", str(args.target),
        "--pop-size", str(args.pop),
        "--generations", str(args.gen),
        "--gravity-steps", str(args.steps),
        "--strategy", strategy,
    ]
    
    if args.fine_tune > 0:
        cmd.extend(["--fine-tune-iters", str(args.fine_tune)])
    
    if args.groups:
        cmd.append("--use-groups")
    
    # Forward any additional flags (e.g., --sa-init-temp, --sa-final-temp, --sa-step-scale)
    cmd.extend(extra_args)
    
    subprocess.run(cmd)
    
    if args.plot and output_file.exists():
        print("\n🎨 Generating visual graph...")
        try:
            from plot_solution import plot_solution
            plot_solution(str(output_file))
        except Exception as e:
            print(f"⚠️ Could not generate plot: {e}")

if __name__ == "__main__":
    main()
