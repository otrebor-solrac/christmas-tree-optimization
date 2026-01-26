use crate::entities::Tree;
use crate::utils::collisions::check_collision_single;
use rand::prelude::*;

pub fn apply_gravity(trees: &mut Vec<Tree>, steps: usize, step_size: f64) {
    let mut rng = thread_rng();
    
    for _ in 0..steps {
        // 1. Calcular Centro de Masa (CoM)
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for t in trees.iter() {
            cx += t.x;
            cy += t.y;
            if t.x < min_x { min_x = t.x; }
            if t.x > max_x { max_x = t.x; }
            if t.y < min_y { min_y = t.y; }
            if t.y > max_y { max_y = t.y; }
        }
        cx /= trees.len() as f64;
        cy /= trees.len() as f64;
        
        // 2. Determinar Factor de Anisotropía (Axis-Awareness)
        let width = max_x - min_x;
        let height = max_y - min_y;
        
        let (force_x, force_y) = if width > height {
            (1.0, 0.2)
        } else {
            (0.2, 1.0)
        };

        // 3. Ordenar por distancia del CENTROIDE al CoM
        let mut indices: Vec<(usize, f64)> = trees.iter().enumerate()
            .map(|(i, t)| {
                use geo::Centroid;
                let c = t.poly.centroid().unwrap();
                let dx = c.x() - cx;
                let dy = c.y() - cy;
                (i, dx*dx + dy*dy)
            })
            .collect();
        
        indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (idx, _) in indices {
            let t = &trees[idx];
            
            use geo::Centroid;
            let c = t.poly.centroid().unwrap();
            let dx = cx - c.x();
            let dy = cy - c.y();
            let dist = (dx*dx + dy*dy).sqrt();

            if dist < 0.001 { continue; }

            let u_x = dx / dist;
            let u_y = dy / dist;
            
            let move_x = u_x * step_size * force_x + rng.gen_range(-0.1..0.1) * step_size;
            let move_y = u_y * step_size * force_y + rng.gen_range(-0.1..0.1) * step_size;

            let old_x = t.x;
            let old_y = t.y;
            let old_angle = t.angle;

            trees[idx].x += move_x;
            trees[idx].y += move_y;
            trees[idx].update_poly();

            // Verificación LOCAL: solo el árbol movido vs los demás (O(n) en lugar de O(n²))
            if check_collision_single(&trees[idx], trees, idx) {
                // Revertir posición
                trees[idx].x = old_x;
                trees[idx].y = old_y;
                trees[idx].update_poly();
                
                // Intentar rotaciones
                let mut best_angle = old_angle;
                let mut min_dist_sq = dist * dist;
                
                let candidates = vec![-45.0, -30.0, -15.0, 15.0, 30.0, 45.0, 90.0, -90.0, 180.0];
                for da in candidates {
                    trees[idx].angle = old_angle + da;
                    trees[idx].update_poly();
                    
                    if !check_collision_single(&trees[idx], trees, idx) {
                        let new_c = trees[idx].poly.centroid().unwrap();
                        let d_sq = (cx - new_c.x()).powi(2) + (cy - new_c.y()).powi(2);
                        if d_sq < min_dist_sq {
                            min_dist_sq = d_sq;
                            best_angle = trees[idx].angle;
                        }
                    }
                }
                
                trees[idx].angle = best_angle;
                trees[idx].update_poly();
            }
        }
    }
}
