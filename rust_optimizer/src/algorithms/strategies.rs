use crate::entities::Tree;
use crate::utils::io::load_trees;
use crate::utils::collisions::check_collisions;
use geo::BoundingRect;
use std::path::Path;

pub fn generate_mosaic(n_target: usize, base_n: usize) -> Option<Vec<Tree>> {
    let base_path = format!("../solutions/T{}.csv", base_n);
    let base_trees = load_trees(Path::new(&base_path)).ok()?;
    
    if base_trees.len() != base_n {
        return None;
    }

    // Calcular bounding box de la base
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in &base_trees {
        if let Some(rect) = t.poly.bounding_rect() {
            min_x = min_x.min(rect.min().x);
            max_x = max_x.max(rect.max().x);
            min_y = min_y.min(rect.min().y);
            max_y = max_y.max(rect.max().y);
        }
    }

    let block_w = max_x - min_x;
    let block_h = max_y - min_y;
    let cx_base = (min_x + max_x) / 2.0;
    let cy_base = (min_y + max_y) / 2.0;

    let mut new_trees = Vec::new();
    let mut id = 1;

    // Grid 2x2
    let cols = 2;
    let rows = 2;
    let start_x = -(cols as f64 * block_w) / 2.0 + block_w / 2.0;
    let start_y = -(rows as f64 * block_h) / 2.0 + block_h / 2.0;

    for r in 0..rows {
        for c in 0..cols {
            let bx = start_x + c as f64 * block_w;
            let by = start_y + r as f64 * block_h;
            
            for t in &base_trees {
                let mut nt = t.clone();
                nt.id = id;
                nt.x = (t.x - cx_base) + bx;
                nt.y = (t.y - cy_base) + by;
                nt.update_poly();
                new_trees.push(nt);
                id += 1;
            }
        }
    }

    if new_trees.len() > n_target {
        new_trees.truncate(n_target);
    }

    // Nota: Con Relate() ya no hay falsos positivos por "toques"
    if check_collisions(&new_trees) {
        return None; // Solo retorna None si hay superposición REAL
    }

    Some(new_trees)
}

pub fn generate_grid_zipper(n_target: usize, stride_x: f64, row_height: f64) -> Option<Vec<Tree>> {
    // Buscar dimensiones óptimas (aspect ratio 1.7 - 2.4)
    let mut best_dims = None;
    let mut min_diff = f64::INFINITY;

    for h in 1..=(n_target as f64).sqrt() as usize {
        if n_target % h == 0 {
            let w = n_target / h;
            let ratio = w as f64 / h as f64;
            if ratio >= 1.7 && ratio <= 2.4 {
                let diff = (ratio - 2.0).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_dims = Some((w, h));
                }
            }
        }
    }

    let (cols, rows) = best_dims?;

    // Parámetros optimizados para N=200: stride_x=0.413, row_height=0.821
    // (Anteriormente 0.412 y 0.85)

    let mut trees = Vec::with_capacity(n_target);
    let mut id = 1;


    for r in 0..rows {
        for c in 0..cols {
            let mut nx = c as f64 * stride_x;
            let mut ny = r as f64 * row_height;
            let mut angle = 0.0;

            if c % 2 != 0 {
                angle = 180.0;
                ny += 0.50;
            }

            if r % 2 != 0 {
                nx -= 0.15;
            }

            let t = Tree::new(id, nx, ny, angle);

            trees.push(t);
            id += 1;
        }
    }

    Some(trees)
}

pub fn generate_pruning(n_target: usize) -> Option<Vec<Tree>> {
    let source_path = format!("solutions/T{}.csv", n_target + 1);
    let trees = load_trees(Path::new(&source_path)).ok()?;
    if trees.len() != n_target + 1 { return None; }
    Some(smart_prune(trees, n_target))
}

pub fn smart_prune(mut trees: Vec<Tree>, n_target: usize) -> Vec<Tree> {
    if trees.len() <= n_target { return trees; }
    
    while trees.len() > n_target {
        // Calcular Centro de Masas actual
        let mut g_cx = 0.0;
        let mut g_cy = 0.0;
        for t in &trees {
            g_cx += t.x;
            g_cy += t.y;
        }
        g_cx /= trees.len() as f64;
        g_cy /= trees.len() as f64;

        let rects: Vec<_> = trees.iter()
            .map(|t| t.poly.bounding_rect().unwrap())
            .collect();

        let mut best_idx = 0;
        let mut min_max_side = f64::INFINITY;
        let mut min_sum_side = f64::INFINITY;
        let mut max_dist_inf = f64::NEG_INFINITY;

        for i in 0..trees.len() {
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;

            for j in 0..trees.len() {
                if i == j { continue; }
                let r = &rects[j];
                if r.min().x < min_x { min_x = r.min().x; }
                if r.max().x > max_x { max_x = r.max().x; }
                if r.min().y < min_y { min_y = r.min().y; }
                if r.max().y > max_y { max_y = r.max().y; }
            }

            let side_w = max_x - min_x;
            let side_h = max_y - min_y;
            let max_s = side_w.max(side_h);
            let sum_s = side_w + side_h;

            // Distancia Chebyshev (L-infinito) al centro de masas
            // Es la distancia ideal para estructuras cuadradas/rectangulares
            let d_x = (trees[i].x - g_cx).abs();
            let d_y = (trees[i].y - g_cy).abs();
            let dist_inf = d_x.max(d_y);

            // CRITERIO DE SELECCIÓN (ONION LAYERS):
            // 1. Reducir el lado máximo de la caja (Kaggle Score)
            if max_s < min_max_side - 1e-9 {
                min_max_side = max_s;
                min_sum_side = sum_s;
                max_dist_inf = dist_inf;
                best_idx = i;
            } 
            else if (max_s - min_max_side).abs() < 1e-9 {
                // 2. DESEMPATE 1: Reducir el perímetro total (mantenerlo compacto)
                if sum_s < min_sum_side - 1e-9 {
                    min_sum_side = sum_s;
                    max_dist_inf = dist_inf;
                    best_idx = i;
                } 
                else if (sum_s - min_sum_side).abs() < 1e-9 {
                    // 3. DESEMPATE 2: Quitar el más lejano al centro (Capa externa)
                    if dist_inf > max_dist_inf {
                        max_dist_inf = dist_inf;
                        best_idx = i;
                    }
                }
            }
        }
        trees.remove(best_idx);
    }
    
    // Re-id
    for (i, t) in trees.iter_mut().enumerate() {
        t.id = i + 1;
        t.update_poly();
    }
    trees
}

pub fn generate_incremental(n_target: usize) -> Option<Vec<Tree>> {
    if n_target == 0 { return None; }
    let source_path = format!("solutions/T{}.csv", n_target - 1);
    let trees = load_trees(Path::new(&source_path)).ok()?;
    if trees.len() != n_target - 1 { return None; }
    
    smart_increment(trees, n_target)
}



pub fn generate_deca_mosaic(target_n: usize) -> Option<Vec<Tree>> {
    let k = target_n / 10;
    let r = target_n % 10;
    let total_cells = k + if r > 0 { 1 } else { 0 };
    
    // Calcular dimension grid (aprox cuadrada)
    let grid_w = (total_cells as f64).sqrt().ceil() as usize;
    
    let mut all_trees = Vec::new();
    // T10 tiene un lado de aprox 2.0. Con 2.1 están casi tocándose.
    let cell_size = 2.1; 
    
    // Centrar la grid en 0,0
    let total_w = grid_w as f64 * cell_size;
    let center_offset = total_w / 2.0 - (cell_size / 2.0);

    for i in 0..total_cells {
        let is_remainder = (i == total_cells - 1) && (r > 0);
        let n_load = if is_remainder { r } else { 10 };
        
        let path = format!("../solutions/T{}.csv", n_load);
        if let Ok(mut trees) = load_trees(Path::new(&path)) {
            // Calcular posición en grid
            let row = i / grid_w;
            let col = i % grid_w;
            
            let offset_x = (col as f64 * cell_size) - center_offset;
            let offset_y = (row as f64 * cell_size) - center_offset;
            
            // PRIMERO: Centrar el bloque en (0,0) calculando su centroide
            let cx: f64 = trees.iter().map(|t| t.x).sum::<f64>() / trees.len() as f64;
            let cy: f64 = trees.iter().map(|t| t.y).sum::<f64>() / trees.len() as f64;
            
            for t in trees.iter_mut() {
                // Centrar en origen, luego mover a posición del grid
                t.x = (t.x - cx) + offset_x;
                t.y = (t.y - cy) + offset_y;
                t.update_poly();
                all_trees.push(t.clone());
            }
        } else {
            println!("⚠️ Error: No se pudo cargar T{}.csv para Deca-Mosaic", n_load);
            return None;
        }
    }
    
    // Renombrar IDs
    for (i, t) in all_trees.iter_mut().enumerate() {
        t.id = i + 1;
    }
    
    if all_trees.len() > target_n {
         all_trees.truncate(target_n);
    }

    Some(all_trees)
}



pub fn smart_increment(mut trees: Vec<Tree>, n_target: usize) -> Option<Vec<Tree>> {
    use crate::utils::collisions::check_collision_single;
    
    if trees.len() >= n_target { return Some(trees); }

    while trees.len() < n_target {
        let current_n = trees.len();
        // Calcular CoM y bounding box
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        
        for t in &trees {
            cx += t.x;
            cy += t.y;
            if t.x < min_x { min_x = t.x; }
            if t.x > max_x { max_x = t.x; }
            if t.y < min_y { min_y = t.y; }
            if t.y > max_y { max_y = t.y; }
        }
        cx /= trees.len() as f64;
        cy /= trees.len() as f64;

        // Probar colocar el nuevo árbol en varios puntos de la periferia
        let offsets = [
            (max_x + 0.5, cy, 0.0),          // Derecha
            (min_x - 0.5, cy, 180.0),        // Izquierda
            (cx, max_y + 0.8, 0.0),          // Arriba
            (cx, min_y - 0.8, 180.0),        // Abajo
            (max_x + 0.3, max_y + 0.5, 0.0), // Esquina superior derecha
            (min_x - 0.3, max_y + 0.5, 180.0), // Esquina superior izquierda
            (max_x + 0.3, min_y - 0.5, 0.0), // Esquina inferior derecha
            (min_x - 0.3, min_y - 0.5, 180.0), // Esquina inferior izquierda
        ];

        let mut placed = false;
        for (nx, ny, angle) in offsets {
            let mut new_tree = Tree::new(current_n + 1, nx, ny, angle);
            new_tree.update_poly();
            
            // Verificar si colisiona con los árboles existentes
            if !check_collision_single(&new_tree, &trees, usize::MAX) {
                trees.push(new_tree);
                placed = true;
                break;
            }
        }
        
        if !placed {
            // Si falla, al menos añadirlo muy lejos para no bloquear el bucle, 
            // aunque idealmente no debería pasar con esta lógica de periferia
            let new_tree = Tree::new(current_n + 1, max_x + 2.0, max_y + 2.0, 0.0);
            trees.push(new_tree);
        }
    }
    Some(trees)
}
