//! Simulated Annealing Optimizer for Christmas Tree Layout
//!
//! Uses temperature-based probabilistic acceptance to escape local minima
//! and explore globally different configurations.

use crate::entities::Tree;
use crate::utils::collisions::check_collision_single;
use crate::utils::geometry::{calculate_score, calculate_kaggle_score, calculate_optimization_score};
use rand::Rng;
use std::path::Path;

/// Optimizes tree layout using Simulated Annealing
///
/// Key difference from CMA-ES: SA can accept WORSE solutions temporarily,
/// allowing it to escape local minima and explore very different configurations.
pub fn optimize_with_annealing(
    mut trees: Vec<Tree>,
    max_iterations: usize,
    verbose: bool,
    freeze_inner: f64, // 0.0 to 1.0 (fraction of inner trees to freeze)
    freeze_outer: f64, // 0.0 to 1.0 (fraction of outer trees to freeze)
    initial_temp: f64,
    final_temp: f64,
    temp_power: f64,
    step_scale: f64,
) -> Vec<Tree> {
    if trees.is_empty() {
        return trees;
    }

    let mut rng = rand::thread_rng();
    let n_trees = trees.len();

    // Temperature parameters
    let cooling_rate: f64 = (final_temp / initial_temp).powf(1.0 / max_iterations as f64);

    // --- ONION PEELING: Determinar índices mutables ---
    // Calcular Centro de Masa (CoM) para clasificar por distancia real al centro del grupo
    let mut com_x = 0.0;
    let mut com_y = 0.0;
    for t in &trees {
        com_x += t.x;
        com_y += t.y;
    }
    com_x /= n_trees as f64;
    com_y /= n_trees as f64;

    let mut mutable_indices = Vec::new();
    
    // Calculamos distancias al Centro de Masa
    let mut dists: Vec<(usize, f64)> = trees.iter()
        .enumerate()
        .map(|(i, t)| (i, (t.x - com_x).hypot(t.y - com_y)))
        .collect();
    
    // Ordenar por distancia (menor a mayor)
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Calcular rangos
    let total = n_trees as f64;
    let start_idx = (total * freeze_inner).floor() as usize;
    let end_idx = (total * (1.0 - freeze_outer)).ceil() as usize;

    for (rank, (idx, _)) in dists.into_iter().enumerate() {
        if rank >= start_idx && rank < end_idx {
            mutable_indices.push(idx);
        }
    }

    if mutable_indices.is_empty() {
        if verbose { println!("⚠️ Todos los árboles están congelados. Saliendo."); }
        return trees;
    }

    if verbose {
        println!("❄️ Congelación activa: {:.1}% Interno, {:.1}% Externo", freeze_inner*100.0, freeze_outer*100.0);
        println!("❄️ Árboles mutables: {} de {}", mutable_indices.len(), n_trees);
    }

    // Movement parameters - scale with layout size
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
    let max_move = spread * 0.3 * step_scale;  // Max 30% of spread per move
    let max_rotate = 45.0 * step_scale;        // Max 45 degrees per move

    // Tracking (Optimization score guides the process)
    let mut current_opt_score = calculate_optimization_score(&trees);
    let mut best_kaggle_score = if crate::utils::collisions::check_collisions(&trees) { f64::INFINITY } else { calculate_kaggle_score(&trees) };
    let mut best_trees = trees.clone();
    let mut temp: f64 = initial_temp;
    
    let mut accepted = 0;
    let mut improved = 0;
    let mut worse_accepted = 0;

    if verbose {
        println!("🔥 [Simulated Annealing] Iniciando optimización:");
        println!("   Árboles: {}", n_trees);
        println!("   Iteraciones: {}", max_iterations);
        println!("   Temp inicial: {:.4}, final: {:.6}", initial_temp, final_temp);
        println!("   Score inicial (Kaggle): {:.6}", best_kaggle_score);
    }

    for iter in 0..max_iterations {
        // Select random tree from MUTABLE list
        let idx = mutable_indices[rng.gen_range(0..mutable_indices.len())];
        
        // Store original position
        let orig_x = trees[idx].x;
        let orig_y = trees[idx].y;
        let orig_angle = trees[idx].angle;

        // Generate perturbation (scale decreases with temperature)
        let scale = temp.powf(temp_power); // Smoother scaling
        
        let dx = rng.gen_range(-max_move..max_move) * scale;
        let dy = rng.gen_range(-max_move..max_move) * scale;
        let da = rng.gen_range(-max_rotate..max_rotate) * scale;

        // Apply perturbation
        trees[idx].x += dx;
        trees[idx].y += dy;
        trees[idx].angle += da;
        trees[idx].update_poly();

        // Check for collisions with this tree only (fast O(n) check)
        let has_collision = check_collision_single(&trees[idx], &trees, idx);

        if has_collision {
            // Revert if collision
            trees[idx].x = orig_x;
            trees[idx].y = orig_y;
            trees[idx].angle = orig_angle;
            trees[idx].update_poly();
        } else {
            // Calculate new score (Internal optimization score)
            let new_opt_score = calculate_optimization_score(&trees);
            let delta = new_opt_score - current_opt_score;

            // Metropolis criterion
            let accept = if delta < 0.0 {
                true // Always accept improvements
            } else {
                // Accept worse with probability exp(-delta/temp)
                let prob = (-delta / temp).exp();
                rng.gen::<f64>() < prob
            };

            if accept {
                current_opt_score = new_opt_score;
                accepted += 1;

                if delta < -1e-12 {
                    improved += 1;
                } else {
                    worse_accepted += 1;
                }

                // Track global best (Kaggle score)
                let new_kaggle_score = calculate_kaggle_score(&trees);
                if new_kaggle_score < best_kaggle_score - 1e-8 {
                    best_kaggle_score = new_kaggle_score;
                    best_trees = trees.clone();
                    if verbose {
                        println!("   ✨ Iter {}: NUEVO RÉCORD (Kaggle: {:.6}) (temp: {:.4})", 
                                 iter, new_kaggle_score, temp);
                    }
                }
            } else {
                // Revert
                trees[idx].x = orig_x;
                trees[idx].y = orig_y;
                trees[idx].angle = orig_angle;
                trees[idx].update_poly();
            }
        }

        // Cool down
        temp *= cooling_rate;

        // Progress report
        if verbose && (iter + 1) % (max_iterations / 10).max(1) == 0 {
            let pct = (iter + 1) * 100 / max_iterations;
            let current_kaggle = calculate_kaggle_score(&trees);
            println!("   {}%: Kaggle {:.6}, Temp {:.6}, Aceptados: {} ({} mejoras, {} peores)",
                     pct, current_kaggle, temp, accepted, improved, worse_accepted);
        }
    }

    // Return best found
    if verbose {
        println!("\n🏆 Simulated Annealing finalizado:");
        println!("   Mejor Kaggle Score: {:.6}", best_kaggle_score);
        println!("   Colisiones: {}", if best_kaggle_score.is_infinite() { "Sí ⚠️" } else { "No ✅" });
        println!("   Total aceptados: {} ({} mejoras, {} peores)", 
                 accepted, improved, worse_accepted);
    }

    best_trees
}

/// Run SA and save if better
pub fn run_annealing_optimizer(
    trees: Vec<Tree>,
    max_iterations: usize,
    output_path: &Path,
) -> Vec<Tree> {
    use crate::utils::io::save_if_better;

    let optimized = optimize_with_annealing(trees, max_iterations, true, 0.0, 0.0, 1.0, 0.001, 0.3, 1.0);
    let score = calculate_score(&optimized);
    
    if score.is_finite() {
        let _ = save_if_better(output_path, &optimized, score);
    }
    
    optimized
}
