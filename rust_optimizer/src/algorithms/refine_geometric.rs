use crate::entities::Tree;
use crate::utils::geometry::{calculate_kaggle_score, calculate_overlap_area};
use crate::utils::io::save_if_better;
use cmaes::{CMAESOptions, DVector};
use std::path::Path;

/// Refina una solución usando el área de solapamiento como guía continua.
/// Esto emula el comportamiento de SciPy (greedy_packer.py) en Rust.
pub fn refine_geometric(
    trees: Vec<Tree>,
    generations: usize,
    output_path: &Path,
) -> Vec<Tree> {
    if trees.is_empty() { return trees; }

    let n_trees = trees.len();
    let dim = n_trees * 3;

    // 1. Preparar parámetros iniciales
    let initial_params: Vec<f64> = trees.iter()
        .flat_map(|t| vec![t.x, t.y, t.angle])
        .collect();

    // 2. Definir Sigma (muy pequeño para refinamiento local)
    let initial_sigma = 0.05; 

    println!("📐 [Refine Geometric] Iniciando refinamiento (N={})", n_trees);
    println!("   Fase: Minimizar Solapamiento + Bounding Box...");

    let objective = |params: &DVector<f64>| -> f64 {
        let mut temp_trees = Vec::with_capacity(n_trees);
        for i in 0..n_trees {
            let x = params[i * 3];
            let y = params[i * 3 + 1];
            let angle = params[i * 3 + 2];
            temp_trees.push(Tree::new(i + 1, x, y, angle));
        }

        let overlap = calculate_overlap_area(&temp_trees);
        let kaggle = calculate_kaggle_score(&temp_trees);

        // Función de costo inspirada en SciPy:
        // Penalizar fuertemente el solapamiento, pero mantener el objetivo del área.
        (overlap * 5000.0) + kaggle
    };

    let mut cmaes_state = CMAESOptions::new(DVector::from_vec(initial_params), initial_sigma)
        .max_generations(generations)
        .population_size(4 + (3.0 * (dim as f64).ln()) as usize)
        .build(objective)
        .unwrap();

    let mut best_trees = trees.clone();
    let mut best_kaggle = calculate_kaggle_score(&best_trees);

    for gen in 0..generations {
        let _ = cmaes_state.next();
        
        if let Some(best_ind) = cmaes_state.current_best_individual() {
            let mut current_trees = Vec::with_capacity(n_trees);
            for i in 0..n_trees {
                current_trees.push(Tree::new(i+1, best_ind.point[i*3], best_ind.point[i*3+1], best_ind.point[i*3+2]));
            }

            let overlap = calculate_overlap_area(&current_trees);
            if overlap < 1e-9 {
                let kaggle = calculate_kaggle_score(&current_trees);
                if kaggle < best_kaggle - 1e-8 {
                    best_kaggle = kaggle;
                    best_trees = current_trees;
                    println!("   ✨ Gen {}: NUEVO RÉCORD (Kaggle: {:.6})", gen, best_kaggle);
                }
            }
        }
    }

    // Guardar si es mejor
    if best_kaggle.is_finite() {
        let _ = save_if_better(output_path, &best_trees, best_kaggle);
    }

    best_trees
}
