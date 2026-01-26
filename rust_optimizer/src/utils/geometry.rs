use crate::entities::{Tree, SCALE_FACTOR};
use crate::utils::collisions::check_collisions;
use geo::{BoundingRect, Area, BooleanOps, Intersects};

/// Calcula el score de Kaggle (Bounding Square Area / N).
/// ... (rest of the file)
/// Formula: max(ancho, alto)^2 / N
/// Retorna inf si hay colisiones.
/// Esta es la función principal que se muestra en la UI y logs.
pub fn calculate_score(trees: &[Tree]) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }

    // Comprobar colisiones primero (permite toques)
    if check_collisions(trees) {
        return f64::INFINITY;
    }

    calculate_kaggle_score(trees)
}

/// Calcula el score de Kaggle basado en el Bounding Square SIN verificar colisiones.
/// Formula: max(ancho, alto)^2 / N
pub fn calculate_kaggle_score(trees: &[Tree]) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees {
        if let Some(rect) = t.poly.bounding_rect() {
            let (t_min_x, t_min_y) = rect.min().x_y();
            let (t_max_x, t_max_y) = rect.max().x_y();
            
            if t_min_x < min_x { min_x = t_min_x; }
            if t_min_y < min_y { min_y = t_min_y; }
            if t_max_x > max_x { max_x = t_max_x; }
            if t_max_y > max_y { max_y = t_max_y; }
        }
    }

    let w = (max_x - min_x) / SCALE_FACTOR;
    let h = (max_y - min_y) / SCALE_FACTOR;
    let side = w.max(h);
    
    (side * side) / (trees.len() as f64)
}

/// Calcula el score de optimización interno.
/// Incluye el score de Kaggle + una penalización por no ser cuadrado.
/// Se usa para guiar algoritmos como GA, CMA-ES y SA.
pub fn calculate_optimization_score(trees: &[Tree]) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }

    let kaggle_score = calculate_kaggle_score(trees);
    
    // Calcular Bounding Box para la penalización de forma
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees {
        if let Some(rect) = t.poly.bounding_rect() {
            let (t_min_x, t_min_y) = rect.min().x_y();
            let (t_max_x, t_max_y) = rect.max().x_y();
            
            if t_min_x < min_x { min_x = t_min_x; }
            if t_min_y < min_y { min_y = t_min_y; }
            if t_max_x > max_x { max_x = t_max_x; }
            if t_max_y > max_y { max_y = t_max_y; }
        }
    }

    let w = (max_x - min_x) / SCALE_FACTOR;
    let h = (max_y - min_y) / SCALE_FACTOR;
    
    // Penalización por diferencia de lados (abs(w - h))
    let square_penalty = (w - h).abs() * 0.1;

    // GRAVEDAD: Penalización pequeña por distancia total al origen
    // Esto ayuda a mantener los árboles compactados hacia el centro.
    let gravity_penalty: f64 = trees.iter()
        .map(|t| (t.x.powi(2) + t.y.powi(2)).sqrt())
        .sum::<f64>() / (trees.len() as f64) * 0.05;
    
    kaggle_score + square_penalty + gravity_penalty
}

/// Mantiene compatibilidad con código que use calculate_score_raw.
/// Calcula el score del bounding box cuadrado (antiguo comportamiento).
pub fn calculate_score_raw(trees: &[Tree]) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees {
        if let Some(rect) = t.poly.bounding_rect() {
            let (t_min_x, t_min_y) = rect.min().x_y();
            let (t_max_x, t_max_y) = rect.max().x_y();
            
            if t_min_x < min_x { min_x = t_min_x; }
            if t_min_y < min_y { min_y = t_min_y; }
            if t_max_x > max_x { max_x = t_max_x; }
            if t_max_y > max_y { max_y = t_max_y; }
        }
    }

    let w = max_x - min_x;
    let h = max_y - min_y;
    let side_unscaled = w.max(h) / SCALE_FACTOR;
    
    (side_unscaled * side_unscaled) / (trees.len() as f64)
}

/// Calcula el área total de solapamiento entre todos los árboles.
/// Se usa para dar un gradiente suave al optimizador geométrico.
pub fn calculate_overlap_area(trees: &[Tree]) -> f64 {
    let mut total_overlap = 0.0;
    
    // Cache bounding boxes
    let bounds: Vec<_> = trees.iter()
        .map(|t| t.poly.bounding_rect().unwrap())
        .collect();

    for i in 0..trees.len() {
        for j in (i + 1)..trees.len() {
            if bounds[i].intersects(&bounds[j]) {
                let inter = trees[i].poly.intersection(&trees[j].poly);
                total_overlap += inter.unsigned_area();
            }
        }
    }
    total_overlap
}

