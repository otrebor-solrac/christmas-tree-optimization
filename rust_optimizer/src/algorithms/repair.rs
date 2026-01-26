use crate::entities::Tree;
use crate::utils::collisions::check_collisions;
use crate::utils::geometry::calculate_score;
use rand::prelude::*;

/// Intenta reparar una configuración inválida (con colisiones) usando múltiples estrategias:
/// 1. Sacudida (Shaking): Perturbaciones aleatorias suaves, medias y fuertes.
/// 2. Expansión (Explosion): Escalar todas las coordenadas alejándolas del origen progresivamente.
pub fn repair_solution(trees: &mut Vec<Tree>, verbose: bool, allow_scaling: bool) -> bool {
    if !check_collisions(trees) {
        return true;
    }

    if verbose {
        println!("⚠️ ALERTA: Configuración con colisiones. Iniciando protocolo de reparación...");
    }

    let mut rng = thread_rng();
    
    // --- ESTRATEGIA 1: SACUDIDA (SHAKING) ---
    // Intentamos resolver colisiones menores vibrando los árboles.
    let shake_strategies = vec![
        (50, 0.0001, 0.1, "Micro"),
        (50, 0.01, 1.0, "Suave"),
        (50, 0.05, 5.0, "Media"),
    ];

    for (iters, pos_range, ang_range, name) in shake_strategies {
        if verbose { println!("   -> Probando Sacudida {}...", name); }
        for _ in 0..iters {
            let mut temp_trees = trees.clone();
            for t in temp_trees.iter_mut() {
                t.x += rng.gen_range(-pos_range..pos_range);
                t.y += rng.gen_range(-pos_range..pos_range);
                t.angle += rng.gen_range(-ang_range..ang_range);
                t.update_poly();
            }

            if !check_collisions(&temp_trees) {
                *trees = temp_trees;
                if verbose { println!("✅ Reparado (Sacudida {})! Score: {:.6}", name, calculate_score(trees)); }
                return true;
            }
        }
    }

    // --- ESTRATEGIA 2: EXPANSIÓN (EXPLOSION) ---
    if !allow_scaling {
        if verbose { println!("❌ No se permiten expansiones. Falló reparación."); }
        return false;
    }

    if verbose { println!("   -> Probando Expansión (Scaling)..."); }
    
    let mut current_scale = 1.0;
    
    // Incrementos progresivos:  
    // Primero intentos finos (1.0 -> 1.5 en pasos de 0.005)
    // Luego intentos más grandes si es necesario
    let step = 0.005; 
    let max_iter = 1000;

    for i in 1..=max_iter {
        current_scale += step; // 1.005, 1.010, ...
        
        let mut temp_trees = trees.clone();
        for t in temp_trees.iter_mut() {
            t.x *= current_scale;
            t.y *= current_scale;
            t.update_poly();
        }

        if !check_collisions(&temp_trees) {
            *trees = temp_trees;
            if verbose { 
                println!("✅ Reparado (Expansión iter {} | x{:.4})! Score: {:.6}", i, current_scale, calculate_score(trees)); 
            }
            return true;
        }
    }

    if verbose { println!("❌ Falló reparación completa. El estado es inválido."); }
    false
}
