use crate::entities::Tree;
use crate::utils::collisions::check_collision_single;
use geo::{ConvexHull, Point, MultiPoint, BoundingRect};
use rand::prelude::*;

pub fn apply_rim_pressure(trees: &mut Vec<Tree>, factor: f64) {
    if trees.is_empty() { return; }

    // 1. Calculate Global Centroid
    let mut cx = 0.0;
    let mut cy = 0.0;
    for t in trees.iter() {
        cx += t.x;
        cy += t.y;
    }
    cx /= trees.len() as f64;
    cy /= trees.len() as f64;
    let _center = Point::new(cx, cy);

    // 2. Compute Convex Hull
    let points: Vec<Point<f64>> = trees.iter()
        .map(|t| Point::new(t.x, t.y))
        .collect();
    let multi_point = MultiPoint::from(points);
    let hull_poly = multi_point.convex_hull();

    // 3. Identify Rim Trees vs Inner Trees
    // The hull returns a Polygon. The exterior coordinates are the points on the hull.
    // We need to match these coordinates back to tree indices to move them.
    // Note: Due to floating point usage, exact matching might be tricky, but we use a small epsilon.
    
    let hull_indices: Vec<usize> = trees.iter().enumerate()
        .filter_map(|(i, t)| {
            let p = Point::new(t.x, t.y);
            // Check if point p is one of the vertices of the hull polygon
            // The hull includes the point itself as a vertex.
            for coord in hull_poly.exterior().coords() {
                let hp = Point::new(coord.x, coord.y);
                if (p.x() - hp.x()).abs() < 1e-6 && (p.y() - hp.y()).abs() < 1e-6 {
                    return Some(i);
                }
            }
            None
        })
        .collect();

    // 4. Apply Pressure
    for &idx in &hull_indices {
        // Calculamos vectores usando referencia inmutable
        let (dx, dy, dist) = {
            let t = &trees[idx];
            let dx = cx - t.x;
            let dy = cy - t.y;
            (dx, dy, (dx*dx + dy*dy).sqrt())
        };

        if dist > 1e-4 {
            let ux = dx / dist;
            let uy = dy / dist;

            // Usamos un candidato (clon) para verificar colisiones sin violar reglas de borrow
            let mut candidate = trees[idx].clone();
            candidate.x += ux * factor;
            candidate.y += uy * factor;
            candidate.update_poly();

            if !check_collision_single(&candidate, trees, idx) {
                trees[idx] = candidate;
            }
        }
    }
}

/// Aplica presión solo en el eje dominante (Ancho o Alto) para "cuadrar" la solución.
/// Esto ataca directamente la métrica de Kaggle: max(W, H).
pub fn apply_square_pressure(trees: &mut Vec<Tree>, step_size: f64) {
    if trees.is_empty() { return; }
    let mut rng = thread_rng();

    // 1. Calcular Bounding Box actual
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees.iter() {
        if let Some(rect) = t.poly.bounding_rect() {
            min_x = min_x.min(rect.min().x);
            max_x = max_x.max(rect.max().x);
            min_y = min_y.min(rect.min().y);
            max_y = max_y.max(rect.max().y);
        }
    }

    let width = max_x - min_x;
    let height = max_y - min_y;
    
    // 2. Determinar dirección de compresión
    // Si es más ancho que alto, comprimimos en X. Si no, en Y.
    let (squeeze_x, squeeze_y) = if width > height {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };

    // 3. Aplicar presión a TODOS los árboles, proporcional a su distancia al centro en ese eje
    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;

    // Ordenamos índices aleatoriamente para no favorecer siempre a los primeros
    let mut indices: Vec<usize> = (0..trees.len()).collect();
    indices.shuffle(&mut rng);

    for idx in indices {
        let (dir_x, dir_y) = {
            let t = &trees[idx];
            let dx = if squeeze_x > 0.0 { cx - t.x } else { 0.0 };
            let dy = if squeeze_y > 0.0 { cy - t.y } else { 0.0 };
            
            if dx.abs() > 0.1 || dy.abs() > 0.1 {
                (
                    if dx.abs() > 1e-4 { dx.signum() } else { 0.0 },
                    if dy.abs() > 1e-4 { dy.signum() } else { 0.0 }
                )
            } else {
                (0.0, 0.0)
            }
        };

        if dir_x != 0.0 || dir_y != 0.0 {
            let mut candidate = trees[idx].clone();
            candidate.x += dir_x * step_size * squeeze_x;
            candidate.y += dir_y * step_size * squeeze_y;
            candidate.update_poly();

            if !check_collision_single(&candidate, trees, idx) {
                trees[idx] = candidate;
            }
        }
    }
}
