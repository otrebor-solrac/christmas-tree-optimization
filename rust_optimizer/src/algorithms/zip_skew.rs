use crate::entities::Tree;
use crate::utils::collisions::check_collisions;
use geo::{BoundingRect, Area, BooleanOps, Relate}; 
use geo::coordinate_position::CoordPos;
use geo::dimensions::Dimensions;
use cmaes::{CMAESOptions, DVector};

/// Parámetros optimizables
#[derive(Clone, Debug)]
pub struct ZipSkewParams {
    pub angle_base: f64,
    pub pair_dx: f64,
    pub pair_dy: f64,
    pub col_step: f64,
    pub row_skew: f64,
    pub row_height: f64,
    pub cols_float: f64, // 7mo parámetro: permite saltar entre dimensiones de grid
}

impl ZipSkewParams {
    pub fn from_slice(slice: &[f64]) -> Self {
        Self {
            angle_base: slice[0],
            pair_dx: slice[1],
            pair_dy: slice[2],
            col_step: slice[3],
            row_skew: slice[4],
            row_height: slice[5],
            cols_float: slice[6],
        }
    }

    pub fn default_warm_start(n: usize) -> Vec<f64> {
        let (ini_cols, _) = calculate_grid_dimensions(n);
        vec![
            10.0,  // angle_base
            0.30,  // pair_dx
            0.70,  // pair_dy
            0.90,  // col_step
            0.30,  // row_skew
            1.10,  // row_height
            ini_cols as f64, // cols_float
        ]
    }
}

const MIN_BOUNDS: [f64; 7] = [-180.0, -2.0, -2.0, 0.1, -2.0, 0.1, 1.0];
const MAX_BOUNDS: [f64; 7] = [ 180.0,  2.0,  2.0, 3.0,  2.0, 3.0, 100.0];

fn clamp_params_array(x: &[f64], n: usize) -> ZipSkewParams {
    let (ini_cols, _) = calculate_grid_dimensions(n);
    let min_cols = (ini_cols as f64 - 12.0).max(1.0);
    let max_cols = (ini_cols as f64 + 20.0).min(100.0);

    ZipSkewParams {
        angle_base: x[0].clamp(MIN_BOUNDS[0], MAX_BOUNDS[0]),
        pair_dx:    x[1].clamp(MIN_BOUNDS[1], MAX_BOUNDS[1]),
        pair_dy:    x[2].clamp(MIN_BOUNDS[2], MAX_BOUNDS[2]),
        col_step:   x[3].clamp(MIN_BOUNDS[3], MAX_BOUNDS[3]),
        row_skew:   x[4].clamp(MIN_BOUNDS[4], MAX_BOUNDS[4]),
        row_height: x[5].clamp(MIN_BOUNDS[5], MAX_BOUNDS[5]),
        cols_float: x[6].clamp(min_cols, max_cols),
    }
}

/// Cálculo de Grid para N=48
/// Prioriza configuraciones exactas (Waste = 0)
pub fn calculate_grid_dimensions(n: usize) -> (usize, usize) {
    let target_rows = ((n as f64) / 4.0).sqrt().round() as usize;
    let search_rows = target_rows.max(1);
    
    let mut best_cols = 1;
    let mut best_rows = (n + 1) / 2;
    let mut best_score = f64::MAX;
    
    // Buscamos grid
    for rows in search_rows.saturating_sub(2)..=search_rows+3 {
        if rows == 0 { continue; }
        // cols necesarias
        let min_cols = (n + (2 * rows) - 1) / (2 * rows);
        
        for cols in min_cols..=(min_cols + 2) {
            let capacity = 2 * cols * rows;
            if capacity < n { continue; }
            
            let ratio = (cols as f64) / (rows as f64);
            let ratio_dist = (ratio - 2.0).abs(); 
            let waste = capacity as f64 - n as f64;
            
            // Penalizamos fuertemente el desperdicio para N=48
            // Para N=48, esto elegirá 6 cols x 4 rows (capacidad 48 exactos)
            let score = ratio_dist + (waste * 2.0); 
            
            if score < best_score {
                best_score = score;
                best_cols = cols;
                best_rows = rows;
            }
        }
    }
    (best_cols, best_rows)
}

pub fn generate_skewed_lattice(
    n: usize,
    params: &ZipSkewParams,
    cols_of_pairs: usize,
    rows: usize,
) -> Vec<Tree> {
    let mut trees = Vec::with_capacity(n);
    let angle_up = params.angle_base;
    let angle_down = params.angle_base + 180.0;
    
    let mut count = 0;
    let mut id = 1;
    
    for r in 0..rows {
        for c in 0..cols_of_pairs {
            if count >= n { break; }
            
            let bx = (c as f64 * params.col_step) + (r as f64 * params.row_skew);
            let by = r as f64 * params.row_height;
            
            trees.push(Tree::new(id, bx, by, angle_up));
            id += 1;
            count += 1;
            
            if count < n {
                let tx = bx + params.pair_dx;
                let ty = by + params.pair_dy;
                trees.push(Tree::new(id, tx, ty, angle_down));
                id += 1;
                count += 1;
            }
        }
    }
    trees
}

fn get_bounds(trees: &[Tree]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees {
        if let Some(rect) = t.poly.bounding_rect() {
            min_x = min_x.min(rect.min().x);
            min_y = min_y.min(rect.min().y);
            max_x = max_x.max(rect.max().x);
            max_y = max_y.max(rect.max().y);
        }
    }
    (min_x, min_y, max_x, max_y)
}

fn objective_function(x: &[f64], n: usize) -> f64 {
    let params = clamp_params_array(x, n);
    let cols_of_pairs = params.cols_float.round().max(1.0) as usize;
    let rows = (n + (2 * cols_of_pairs) - 1) / (2 * cols_of_pairs);

    let trees = generate_skewed_lattice(n, &params, cols_of_pairs, rows);
    let (min_x, min_y, max_x, max_y) = get_bounds(&trees);
    
    // Score principal (Bounding Box Lado Máximo)
    let score = (max_x - min_x).max(max_y - min_y);

    // --- PENALIZACIÓN HÍBRIDA (Matching Python Logic) ---
    let mut total_overlap_area = 0.0;
    let mut strict_collision_found = false;
    
    // Al ser un grid, solo necesitamos revisar vecinos cercanos para ser eficientes
    // En N grande, (i+30) es suficiente para cubrir vecindarios inmediatos en 2D
    for i in 0..trees.len() {
        let check_limit = (i + 30).min(trees.len()); 
        for j in (i + 1)..check_limit {
            let poly_i = &trees[i].poly;
            let poly_j = &trees[j].poly;
            
            // 1. Penalización suave por área (para dar gradiente)
            let intersection = poly_i.intersection(poly_j);
            let area = intersection.unsigned_area();
            if area > 1e-8 {
                total_overlap_area += area;
            }

            // 2. Penalización dura por colisión estricta (Kaggle: Intersect AND NOT Touch)
            // Usamos Relate para ser consistentes con el visualizador
            if !strict_collision_found {
                let matrix = poly_i.relate(poly_j);
                if matrix.get(CoordPos::Inside, CoordPos::Inside) != Dimensions::Empty {
                    strict_collision_found = true;
                }
            }
        }
    }

    let mut penalty = 0.0;
    if total_overlap_area > 1e-7 {
        penalty += 20.0 + (total_overlap_area * 1000.0);
    }
    if strict_collision_found {
        penalty += 50.0;
    }

    score + penalty
}

pub fn optimize_zip_skew(n: usize, max_generations: usize, verbose: bool) -> Option<Vec<Tree>> {
    // 1. Configurar CMA-ES (7 parámetros ahora)
    let initial_values = ZipSkewParams::default_warm_start(n);
    let sigma = 0.33; // Sigma más alto como en Python para saltar entre enteros de 'cols'
    let initial_dvec = DVector::from_vec(initial_values);
    
    let cmaes_opts = CMAESOptions::new(initial_dvec, sigma)
        .population_size(80) // Aumentamos población para los 7 parámetros
        .max_generations(max_generations);

    // 2. Ejecutar
    let objective = |x: &DVector<f64>| -> f64 {
        objective_function(x.as_slice(), n)
    };
    
    let mut cmaes_state = match cmaes_opts.build(objective) {
        Ok(s) => s,
        Err(e) => {
            println!("Error CMA-ES: {:?}", e);
            return None;
        }
    };

    let mut best_score = f64::MAX;
    let mut best_params_vec = ZipSkewParams::default_warm_start(n);
    let mut stagnation = 0;

    for gen in 0..max_generations {
        let _result = cmaes_state.next();
        
        if let Some(best) = cmaes_state.current_best_individual() {
            let score = best.value;
            
            if score < best_score - 1e-7 {
                best_score = score;
                best_params_vec = best.point.as_slice().to_vec();
                stagnation = 0;
                
                if verbose && gen % 20 == 0 {
                    println!("[ZipSkew] Gen {}: score={:.6}", gen, score);
                }
            } else {
                stagnation += 1;
            }
        }
        
        if stagnation >= 150 {
            if verbose { println!("[ZipSkew] Convergencia en gen {}", gen); }
            break;
        }
    }

    let final_params = clamp_params_array(&best_params_vec, n);
    let final_cols = final_params.cols_float.round().max(1.0) as usize;
    let final_rows = (n + (2 * final_cols) - 1) / (2 * final_cols);

    if verbose {
        println!("[ZipSkew] Params Finales: Grid={}x{}, Skew={:.4}, Height={:.4}", 
            final_cols, final_rows, final_params.row_skew, final_params.row_height);
    }

    let trees = generate_skewed_lattice(n, &final_params, final_cols, final_rows);
    
    // SIEMPRE retornar árboles - Gravity y Fine-Tuning arreglarán colisiones
    if verbose && check_collisions(&trees) {
        println!("[ZipSkew] ⚠️ Hay colisiones menores, serán corregidas por refinamiento.");
    }
    Some(trees)
}