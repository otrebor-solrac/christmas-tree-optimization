use crate::entities::Tree;
use geo::prelude::*;
use geo::coordinate_position::CoordPos;
use geo::dimensions::Dimensions;

/// Comprueba colisiones entre árboles.
/// Usa geo::Relate para detectar SOLO superposición de interiores.
/// Los árboles que se tocan en el borde (sin superponerse) son VÁLIDOS.
pub fn check_collisions(trees: &[Tree]) -> bool {
    let bounds: Vec<_> = trees.iter()
        .map(|t| t.poly.bounding_rect().unwrap())
        .collect();

    for i in 0..trees.len() {
        for j in (i + 1)..trees.len() {
            if bounds[i].intersects(&bounds[j]) {
                // Usar Relate para verificar si los INTERIORES se superponen
                let matrix = trees[i].poly.relate(&trees[j].poly);
                if matrix.get(CoordPos::Inside, CoordPos::Inside) != Dimensions::Empty {
                    return true; // Superposición real
                }
            }
        }
    }
    false
}

pub fn check_collision_single(target: &Tree, others: &[Tree], ignore_idx: usize) -> bool {
    let target_bounds = target.poly.bounding_rect().unwrap();

    for (i, other) in others.iter().enumerate() {
        if i == ignore_idx { continue; }
        if target_bounds.intersects(&other.poly.bounding_rect().unwrap()) {
            let matrix = target.poly.relate(&other.poly);
            if matrix.get(CoordPos::Inside, CoordPos::Inside) != Dimensions::Empty {
                return true;
            }
        }
    }
    false
}
