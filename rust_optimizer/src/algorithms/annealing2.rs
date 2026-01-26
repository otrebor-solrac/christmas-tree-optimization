use crate::entities::Tree;
use crate::utils::collisions::{check_collision_single, check_collisions};
use crate::utils::geometry::{calculate_kaggle_score};
use crate::utils::io::save_solution;
use rand::Rng;
use std::f64;

/// Optimizes tree layout using Enhanced Simulated Annealing
/// Features: Adaptive Steps, Chebyshev Gravity, Swaps, and Flips.
pub fn optimize_with_annealing(
    mut trees: Vec<Tree>,
    max_iterations: usize,
    verbose: bool,
    freeze_inner: f64, // 0.0 to 1.0
    freeze_outer: f64, // 0.0 to 1.0
    initial_temp: f64,
    final_temp: f64,
    _temp_power: f64, // Ignorado en esta versión adaptativa
    base_step_scale: f64,
    live_file: Option<std::path::PathBuf>,
) -> Vec<Tree> {
    if trees.is_empty() {
        return trees;
    }

    let mut rng = rand::thread_rng();
    let n_trees = trees.len();

    // --- 1. CONFIGURACIÓN DEL ONION PEELING ---
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

    // --- 2. PREPARACIÓN DE MÉTRICAS ---
    // Calculamos el bounding box inicial para escalar movimientos
    let mut max_extent: f64 = 1.0;
    for t in &trees {
        max_extent = max_extent.max(t.x.abs()).max(t.y.abs());
    }
    
    // Variables de control adaptativo
    let mut current_step_scale = base_step_scale;
    let mut acceptance_rate_window = 0;
    let mut accepted_in_window = 0;
    
    // Contadores globales
    let mut total_accepted = 0;
    let mut total_improved = 0;
    let mut total_worse = 0;
    
    let check_interval = 100;

    // Estado inicial
    let mut current_energy = get_cost_chebyshev(&trees);
    let mut best_kaggle_score = if check_collisions(&trees) { f64::INFINITY } else { calculate_kaggle_score(&trees) };
    let mut best_trees = trees.clone();
    let mut temp = initial_temp;
    let cooling_rate = (final_temp / initial_temp).powf(1.0 / max_iterations as f64);

    if verbose {
        println!("🔥 [SA Pro] Start: {} trees. Mutable: {}", n_trees, mutable_indices.len());
        println!("   Temp: {:.4} -> {:.8}", initial_temp, final_temp);
        println!("   Initial Score: {:.6}", best_kaggle_score);
    }

    // --- 3. BUCLE PRINCIPAL ---
    for iter in 0..max_iterations {
        // --- ESTRATEGIA DE MOVIMIENTO ---
        // 5% Swap, 10% Flip 180, 85% Move/Rotate normal
        let move_type = rng.gen_range(0..100);
        
        // Guardar estado para revertir (Undo Log)
        let backup_indices: Vec<usize>; // Índices modificados
        let backup_trees: Vec<Tree>;    // Copia de seguridad de esos árboles

        let mut collision_detected = false;

        if move_type < 5 && mutable_indices.len() > 1 {
            // === SWAP (Intercambio de posición) ===
            let idx1 = mutable_indices[rng.gen_range(0..mutable_indices.len())];
            let mut idx2 = mutable_indices[rng.gen_range(0..mutable_indices.len())];
            while idx1 == idx2 { idx2 = mutable_indices[rng.gen_range(0..mutable_indices.len())]; }

            backup_indices = vec![idx1, idx2];
            backup_trees = vec![trees[idx1].clone(), trees[idx2].clone()];

            // Swap coordenadas, mantener rotación
            let temp_x = trees[idx1].x;
            let temp_y = trees[idx1].y;
            trees[idx1].x = trees[idx2].x;
            trees[idx1].y = trees[idx2].y;
            trees[idx2].x = temp_x;
            trees[idx2].y = temp_y;
            
            trees[idx1].update_poly();
            trees[idx2].update_poly();

            // Check colisiones en ambos
            if check_collision_single(&trees[idx1], &trees, idx1) || 
               check_collision_single(&trees[idx2], &trees, idx2) {
                collision_detected = true;
            }

        } else {
            // === PERTURBACIÓN ESTÁNDAR ===
            // (Eliminado el Flip 180 por petición del usuario para evitar giros bruscos)
            let idx = mutable_indices[rng.gen_range(0..mutable_indices.len())];
            backup_indices = vec![idx];
            backup_trees = vec![trees[idx].clone()];

            let spread = max_extent * 0.1 * current_step_scale;
            // Limitamos a 45 grados máximo de desviación (angle_spread)
            let angle_spread = (30.0 * current_step_scale).min(45.0);

            trees[idx].x += rng.gen_range(-spread..spread);
            trees[idx].y += rng.gen_range(-spread..spread);
            trees[idx].angle += rng.gen_range(-angle_spread..angle_spread);
            trees[idx].update_poly();

            if check_collision_single(&trees[idx], &trees, idx) {
                collision_detected = true;
            }
        }

        // --- 4. EVALUACIÓN Y ACEPTACIÓN ---
        if collision_detected {
            // Revertir inmediatamente si hay colisión (Hard Constraint)
            for (i, &tree_idx) in backup_indices.iter().enumerate() {
                trees[tree_idx] = backup_trees[i].clone();
            }
        } else {
            // Calcular nueva energía (Chebyshev para cuadrados)
            let new_energy = get_cost_chebyshev(&trees);
            let delta = new_energy - current_energy;

            // Criterio de Metrópolis
            let accept = if delta < 0.0 {
                true
            } else {
                let prob = (-delta / temp).exp();
                rng.gen::<f64>() < prob
            };

            if accept {
                current_energy = new_energy;
                accepted_in_window += 1;
                total_accepted += 1;
                if delta < 0.0 { total_improved += 1; } else { total_worse += 1; }

                // Chequear si es récord global (Kaggle Score real)
                // Hacemos esto menos frecuentemente o si la energía bajó mucho
                if delta < 0.0 {
                    let kaggle_score = calculate_kaggle_score(&trees);
                    if kaggle_score < best_kaggle_score {
                        best_kaggle_score = kaggle_score;
                        best_trees = trees.clone();
                        if verbose {
                            print!("\r   ✨ New Best: {:.6} (Iter {})    ", best_kaggle_score, iter);
                        }
                    }
                }
            } else {
                // Revertir cambio rechazado
                for (i, &tree_idx) in backup_indices.iter().enumerate() {
                    trees[tree_idx] = backup_trees[i].clone();
                }
            }
        }

        if max_iterations >= 10 && iter % (max_iterations / 10) == 0 {
            let progress = (iter as f64 / max_iterations as f64) * 100.0;
            if verbose {
                println!("   {:.0}%: Kaggle {:.6}, Temp {:.6}, Aceptados: {} ({} mejoras, {} peores)", 
                         progress, best_kaggle_score, temp, total_accepted, total_improved, total_worse);
            }
        }

        // Snapshot visual frecuente (cada 2000 iteraciones)
        // Guardamos el estado ACTUAL (trees) solo si se especificó un archivo
        if iter % 2000 == 0 {
            if let Some(ref path) = live_file {
                let _ = save_solution(path, &trees, current_energy);
            }
        }

        // --- 5. MANTENIMIENTO (Adaptative Step & Cooling) ---
        acceptance_rate_window += 1;
        if acceptance_rate_window >= check_interval {
            let rate = accepted_in_window as f64 / check_interval as f64;
            
            // Ajustar paso para mantener tasa de aceptación saludable (20% - 40%)
            if rate > 0.4 {
                current_step_scale *= 1.05; // Aumentar agresividad
            } else if rate < 0.15 {
                current_step_scale *= 0.90; // Reducir agresividad
            }
            
            // Clamp
            current_step_scale = current_step_scale.clamp(0.01, 2.0);

            acceptance_rate_window = 0;
            accepted_in_window = 0;
        }

        temp *= cooling_rate;
    }

    if verbose {
        println!("\n🏆 SA Finalizado. Score: {:.6}", best_kaggle_score);
        println!("   Aceptados: {} (Mejoras: {}, Peores: {})", total_accepted, total_improved, total_worse);
    }

    best_trees
}

/// Cost Function: Chebyshev Distance (Square Gravity)
/// Penaliza fuertemente el lado máximo (max_x, max_y) para forzar un cuadrado.
/// También incluye una pequeña atracción al centro para compactar.
fn get_cost_chebyshev(trees: &Vec<Tree>) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut sum_dist_sq = 0.0;

    for t in trees {
        // Bounding Box (usando la nueva función helper)
        let (tx_min, tx_max, ty_min, ty_max) = get_tree_bounds(t);
        
        if tx_min < min_x { min_x = tx_min; }
        if tx_max > max_x { max_x = tx_max; }
        if ty_min < min_y { min_y = ty_min; }
        if ty_max > max_y { max_y = ty_max; }

        // Atracción central suave
        sum_dist_sq += t.x*t.x + t.y*t.y;
    }

    let width = max_x - min_x;
    let height = max_y - min_y;
    
    // LA CLAVE: Max(Width, Height) es lo que Kaggle evalúa.
    // Alineamos la energía con el Score de Kaggle: side^2 / N
    // Y añadimos un pequeño tie-breaker para compactación.
    let side = width.max(height); // Asumiendo SCALE_FACTOR = 1.0 como en entities.rs
    let n = trees.len() as f64;
    let kaggle_term = (side * side) / n;
    
    // Costo = KaggleScore + (Compactación / N / 1M)
    kaggle_term + (sum_dist_sq / n) * 0.000001
}

// Helper simple para calcular los bounds de un Tree sin modificar la struct Tree
fn get_tree_bounds(t: &Tree) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    
    for p in t.poly.exterior().coords() {
        if p.x < min_x { min_x = p.x; }
        if p.x > max_x { max_x = p.x; }
        if p.y < min_y { min_y = p.y; }
        if p.y > max_y { max_y = p.y; }
    }
    (min_x, max_x, min_y, max_y)
}
