// CSV Input/Output operations
use crate::entities::Tree;
use crate::utils::collisions::check_collisions;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CsvRow {
    id: usize,
    x: String,
    y: String,
    deg: String,
    score: String,
}

pub fn load_trees(path: &Path) -> Result<Vec<Tree>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut trees = Vec::new();
    for result in rdr.deserialize() {
        let record: CsvRow = result?;
        let x = record.x.trim_start_matches('s').parse::<f64>()?;
        let y = record.y.trim_start_matches('s').parse::<f64>()?;
        let deg = record.deg.trim_start_matches('s').parse::<f64>()?;
        trees.push(Tree::new(record.id, x, y, deg));
    }
    Ok(trees)
}

pub fn save_solution(path: &Path, trees: &[Tree], score: f64) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(&["id", "x", "y", "deg", "score"])?;
    
    for t in trees {
        wtr.write_record(&[
            t.id.to_string(),
            format!("s{:.18}", t.x),
            format!("s{:.18}", t.y),
            format!("s{:.18}", t.angle),
            format!("s{:.18}", score),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn get_score_from_file(path: &Path) -> f64 {
    if !path.exists() { return f64::INFINITY; }
    let rdr_res = csv::Reader::from_path(path);
    let mut rdr = match rdr_res {
        Ok(r) => r,
        Err(_) => return f64::INFINITY,
    };
    for result in rdr.deserialize() {
        let record: CsvRow = match result {
            Ok(r) => r,
            Err(_) => return f64::INFINITY,
        };
        return record.score.trim_start_matches('s').parse::<f64>().unwrap_or(f64::INFINITY);
    }
    f64::INFINITY
}

pub fn save_if_better(path: &Path, trees: &[Tree], new_score: f64) -> Result<bool, Box<dyn Error>> {
    let new_has_collisions = check_collisions(trees);
    
    // Verificar si el archivo actual existe y obtener su estado
    let (old_score, old_has_collisions) = if path.exists() {
        if let Ok(old_trees) = load_trees(path) {
            let has_col = check_collisions(&old_trees);
            (get_score_from_file(path), has_col)
        } else {
            (f64::INFINITY, true) // Error al cargar = tratar como inválido
        }
    } else {
        (f64::INFINITY, true) // No existe
    };
    
    // Decidir si guardar:
    // 1. Si ambas son válidas (sin colisiones) → comparar scores normalmente
    // 2. Si la nueva tiene colisiones pero mejor score raw → guardar para arreglar después
    // 3. Si la vieja tiene colisiones y la nueva no → reemplazar
    
    let should_save = if !new_has_collisions && !old_has_collisions {
        // Ambas válidas: comparar scores
        if new_score < old_score - 1e-7 {
            println!("✨ ¡Mejora! {:.6} -> {:.6}", old_score, new_score);
            true
        } else {
            false
        }
    } else if !new_has_collisions && old_has_collisions {
        // Nueva válida, vieja inválida: reemplazar
        println!("🔄 Reemplazando solución con colisiones por válida (score: {:.6})", new_score);
        true
    } else if new_has_collisions && old_has_collisions {
        // Ambas con colisiones: guardar si mejor score raw
        if new_score < old_score - 1e-7 {
            println!("📝 Guardando mejor layout con colisiones (para arreglar después): {:.6}", new_score);
            true
        } else {
            false
        }
    } else {
        // Nueva con colisiones, vieja válida: NO reemplazar válida por inválida
        false
    };
    
    if should_save {
        save_solution(path, trees, new_score)?;
        return Ok(true);
    }
    
    Ok(false)
}

