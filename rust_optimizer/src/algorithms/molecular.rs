use crate::entities::Tree;
use crate::utils::collisions::check_collision_single;
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct Force {
    fx: f64,
    fy: f64,
}

pub fn apply_molecular_dynamics(trees: &mut Vec<Tree>, steps: usize, step_size: f64) {
    let n = trees.len();
    let mut rng = thread_rng();
    
    // Configuración de física - Ajustada para árboles de w=0.7, h=1.0
    let repulsion_dist = 0.8; // Distancia de interacción (aprox "diámetro" efectivo)
    let repulsion_strength = 1.0; // Fuerza repulsiva moderada
    let gravity_strength = 0.1; // Gravedad central aumentada ligeramente
    let temperature = 0.05; // Ruido térmico inicial menor

    for step in 0..steps {
        if step % (steps / 10).max(1) == 0 {
             println!("   ⚛️  Step {}/{}", step, steps);
        }
        
        // Pseudo-recocido para la "temperatura" (vibración)
        let current_temp = temperature * (1.0 - step as f64 / steps as f64);
        
        // 1. Calcular Centro de Masa (CoM) para Gravedad
        let mut cx = 0.0;
        let mut cy = 0.0;
        for t in trees.iter() {
            cx += t.x;
            cy += t.y;
        }
        cx /= n as f64;
        cy /= n as f64;

        // 2. Calcular Fuerzas (Paralelo)
        // Usamos una copia inmutable para leer posiciones
        let positions: Vec<(f64, f64)> = trees.iter().map(|t| (t.x, t.y)).collect();
        
        let forces: Vec<Force> = (0..n).into_par_iter().map(|i| {
            let (ix, iy) = positions[i];
            let mut fx = 0.0;
            let mut fy = 0.0;

            // A. Fuerza Repulsiva (Interacción pares)
            for j in 0..n {
                if i == j { continue; }
                let (jx, jy) = positions[j];
                let dx = ix - jx;
                let dy = iy - jy;
                let dist_sq = dx*dx + dy*dy;
                
                // Solo vecinos cercanos
                if dist_sq < repulsion_dist * repulsion_dist && dist_sq > 0.000001 {
                    let dist = dist_sq.sqrt();
                    // Ley de repulsión inversa suave: F ~ 1/d
                    let force = repulsion_strength * (1.0 - dist / repulsion_dist);
                    fx += (dx / dist) * force;
                    fy += (dy / dist) * force;
                }
            }

            // B. Gravedad Central (Mantener cohesión)
            let dx_c = cx - ix;
            let dy_c = cy - iy;
            fx += dx_c * gravity_strength;
            fy += dy_c * gravity_strength;

            Force { fx, fy }
        }).collect();

        // 3. Aplicar Movimiento e intentar validar
        for i in 0..n {
            let f = forces[i];
            
            // Ruido térmico
            let noise_x = rng.gen_range(-1.0..1.0) * current_temp;
            let noise_y = rng.gen_range(-1.0..1.0) * current_temp;

            let move_x = (f.fx * step_size) + noise_x;
            let move_y = (f.fy * step_size) + noise_y;

            // Guardar estado previo
            let old_x = trees[i].x;
            let old_y = trees[i].y;

            // Mover
            trees[i].x += move_x;
            trees[i].y += move_y;
            trees[i].update_poly();

            // Validación "blanda": Si choca, intentamos rotar. Si sigue chocando, revertimos PARCIALMENTE
            // para permitir cierta "presión" que se resuelva en siguientes frames.
            if check_collision_single(&trees[i], trees, i) {
                // Intentar micro-rotaciones rápidas para aliviar tensión
                let best_angle = trees[i].angle;
                let mut resolved = false;
                
                // Rotaciones locales
                for da in [-5.0, 5.0, -15.0, 15.0] {
                    trees[i].angle = best_angle + da;
                    trees[i].update_poly();
                    if !check_collision_single(&trees[i], trees, i) {
                        resolved = true;
                        break;
                    }
                }

                if !resolved {
                    // Si no se resuelve rotando, revertimos la translación pero conservamos el mejor ángulo intentado (si hubo)
                    // O mejor: revertir a posición segura
                    trees[i].angle = best_angle;
                    trees[i].x = old_x; 
                    trees[i].y = old_y;
                    trees[i].update_poly();
                }
            }
        }
    }
}
