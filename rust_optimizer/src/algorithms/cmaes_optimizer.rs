//! CMA-ES Optimizer for Christmas Tree Layout
//! 
//! Uses Covariance Matrix Adaptation Evolution Strategy to optimize
//! tree positions and rotations to minimize the bounding square area.
//! Features adaptive sigma that increases when optimization stagnates.

use crate::entities::Tree;
use crate::utils::collisions::check_collisions;
use crate::utils::geometry::{calculate_score, calculate_kaggle_score, calculate_optimization_score};
use cmaes::{CMAESOptions, DVector};
use std::path::Path;

/// Optimizes tree layout using CMA-ES algorithm with adaptive sigma
/// 
/// When optimization stagnates, sigma is increased to allow escaping local minima.
pub fn optimize_with_cmaes(
    mut trees: Vec<Tree>,
    max_generations: usize,
    pop_size: usize,
    stagnation_limit: usize,
    verbose: bool,
) -> Vec<Tree> {
    if trees.is_empty() {
        return trees;
    }

    let n_trees = trees.len();
    let dim = n_trees * 3;

    // Calculate spread for sigma scaling
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for t in &trees {
        if t.x < min_x { min_x = t.x; }
        if t.x > max_x { max_x = t.x; }
        if t.y < min_y { min_y = t.y; }
        if t.y > max_y { max_y = t.y; }
    }
    let spread = ((max_x - min_x).max(max_y - min_y)).max(1.0);

    // Adaptive sigma parameters
    let sigma_levels = [0.05, 0.15, 0.30, 0.50]; // 5%, 15%, 30%, 50% of spread
    let max_restarts = sigma_levels.len();

    // Global best tracking (based on optimization score for guidance)
    let mut global_best_opt_score = calculate_optimization_score(&trees);
    let mut global_best_kaggle_score = if check_collisions(&trees) { f64::INFINITY } else { calculate_kaggle_score(&trees) };
    let mut global_best_params: Vec<f64> = trees.iter()
        .flat_map(|t| vec![t.x, t.y, t.angle])
        .collect();

    if verbose {
        println!("🧬 [CMA-ES Adaptativo] Iniciando optimización:");
        println!("   Dimensiones: {} ({}x3 parámetros)", dim, n_trees);
        println!("   Max generaciones por restart: {}", max_generations / max_restarts);
        println!("   Pop Size: {} | Stagnation Limit: {}", 
                 if pop_size > 0 { pop_size.to_string() } else { "Auto".to_string() }, 
                 stagnation_limit);
        println!("   Score inicial (Kaggle): {:.6}", global_best_kaggle_score);
    }

    for (restart_idx, &sigma_pct) in sigma_levels.iter().enumerate() {
        let initial_sigma = spread * sigma_pct;
        let gens_per_restart = max_generations / max_restarts;

        if verbose {
            println!("\n🔄 Restart {}/{}: Sigma = {:.2}% ({:.4})", 
                     restart_idx + 1, max_restarts, sigma_pct * 100.0, initial_sigma);
        }

        // Start from global best
        let initial_dvec = DVector::from_vec(global_best_params.clone());

        let objective = |params: &DVector<f64>| -> f64 {
            let mut temp_trees: Vec<Tree> = Vec::with_capacity(n_trees);
            for i in 0..n_trees {
                let x = params[i * 3];
                let y = params[i * 3 + 1];
                let angle = params[i * 3 + 2];
                temp_trees.push(Tree::new(i + 1, x, y, angle));
            }
            if check_collisions(&temp_trees) {
                f64::INFINITY
            } else {
                calculate_optimization_score(&temp_trees)
            }
        };

        let mut cmaes_opts = CMAESOptions::new(initial_dvec, initial_sigma)
            .max_generations(gens_per_restart);
            
        if pop_size > 0 {
             cmaes_opts = cmaes_opts.population_size(pop_size);
        } else {
             cmaes_opts = cmaes_opts.population_size(4 + (3.0 * (dim as f64).ln()) as usize);
        }

        let mut cmaes_state = match cmaes_opts.build(objective) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let mut stagnation = 0;
        let mut local_best_opt_score = global_best_opt_score;

        for gen in 0..gens_per_restart {
            let result = cmaes_state.next();

            if let Some(best) = cmaes_state.current_best_individual() {
                let score = best.value;
                
                if score < local_best_opt_score - 1e-8 {
                    local_best_opt_score = score;
                    stagnation = 0;
                    
                    if score < global_best_opt_score - 1e-8 {
                        global_best_opt_score = score;
                        global_best_params = best.point.iter().cloned().collect();
                        
                        // Reconstuir temporalmente para calcular el Kaggle score
                        let mut best_trees = Vec::with_capacity(n_trees);
                        for i in 0..n_trees {
                            best_trees.push(Tree::new(i + 1, global_best_params[i * 3], global_best_params[i * 3 + 1], global_best_params[i * 3 + 2]));
                        }
                        global_best_kaggle_score = calculate_kaggle_score(&best_trees);

                        if verbose {
                            println!("   ✨ Gen {}: NUEVO RÉCORD (Kaggle: {:.6})", gen, global_best_kaggle_score);
                        }
                    }
                } else {
                    stagnation += 1;
                }
            }

            if stagnation >= stagnation_limit || result.is_some() {
                if verbose && stagnation >= stagnation_limit {
                    println!("   ⏸️  Estancado en gen {}. Próximo restart con sigma mayor...", gen);
                }
                break;
            }
        }
    }

    // Reconstruct best trees
    for i in 0..n_trees {
        trees[i].x = global_best_params[i * 3];
        trees[i].y = global_best_params[i * 3 + 1];
        trees[i].angle = global_best_params[i * 3 + 2];
        trees[i].update_poly();
    }

    if verbose {
        let final_score = calculate_score(&trees);
        let has_collisions = check_collisions(&trees);
        println!("\n🏆 CMA-ES Adaptativo finalizado:");
        println!("   Score final: {:.6}", final_score);
        println!("   Colisiones: {}", if has_collisions { "Sí ⚠️" } else { "No ✅" });
    }

    trees
}

/// Run CMA-ES and save if better
pub fn run_cmaes_optimizer(
    trees: Vec<Tree>,
    max_generations: usize,
    output_path: &Path,
) -> Vec<Tree> {
    use crate::utils::io::save_if_better;

    let optimized = optimize_with_cmaes(trees, max_generations, 0, 50, true);
    let score = calculate_score(&optimized);
    
    if score.is_finite() {
        let _ = save_if_better(output_path, &optimized, score);
    }
    
    optimized
}
