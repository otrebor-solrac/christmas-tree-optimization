use crate::entities::Tree;
use crate::utils::collisions::check_collision_single;
use crate::utils::geometry::{calculate_score, calculate_optimization_score};
use rand::prelude::*;
use crate::algorithms::repair::repair_solution;

pub fn fine_tuning(mut trees: Vec<Tree>, iterations: usize, verbose: bool) -> Vec<Tree> {
    if verbose {
        println!("🔧 Iniciando Fine-Tuning (Rust) por {} iteraciones...", iterations);
    }
    let mut rng = thread_rng();

    // 1. Verificación Inicial y Reparación
    if !repair_solution(&mut trees, verbose, true) {
        if verbose { println!("❌ Falló reparación inicial en Fine-Tuning."); }
        return trees;
    }

    let mut best_opt_score = calculate_optimization_score(&trees);
    if verbose { println!("🔹 Score Base (Kaggle): {:.6}", calculate_score(&trees)); }

    // Parámetros de recocido
    let mut step_pos = 0.025;
    let mut step_rot = 0.05;
    let decay = 0.9995;

    for i in 0..iterations {
        // Seleccionar un árbol para perturbar
        let idx = rng.gen_range(0..trees.len());
        
        // Guardar estado original del árbol
        let original_tree = trees[idx].clone();
        
        // Perturbar
        {
            let t = &mut trees[idx];
            t.x += (rng.gen::<f64>() - 0.5) * step_pos;
            t.y += (rng.gen::<f64>() - 0.5) * step_pos;
            t.angle += (rng.gen::<f64>() - 0.5) * step_rot;
            t.update_poly();
        }

        // Verificación LOCAL: solo el árbol modificado vs los demás (O(n) en lugar de O(n²))
        let valid = !check_collision_single(&trees[idx], &trees, idx);
        
        let mut revert = true;
        if valid {
            let new_opt_score = calculate_optimization_score(&trees);
            
            if new_opt_score < best_opt_score {
                best_opt_score = new_opt_score;
                if verbose { println!("✨ Iter {}: Récord! Kaggle Score: {:.7}", i, calculate_score(&trees)); }
                revert = false;
            }
        }
        
        if revert {
            trees[idx] = original_tree;
        }

        // Configuración de paso (Enfriamiento/Recalentamiento)
        step_pos *= decay;
        step_rot *= decay;
        
        if step_pos < 0.00001 {
            step_pos = 0.001;
            step_rot = 0.1;
        }
    }
    
    trees
}
