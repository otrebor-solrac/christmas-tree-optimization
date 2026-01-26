use crate::entities::Tree;
use crate::utils::geometry::{calculate_kaggle_score, calculate_overlap_area};
use rand::Rng;

/// Configuración para el Soft SA (Legacy / Standalone)
pub struct SoftSAConfig {
    pub max_iterations: usize,
    pub initial_temp: f64,
    pub cooling_rate: f64,
    pub penalty_factor_start: f64,
    pub penalty_factor_end: f64,
    pub sa_step_scale: f64,
}

impl Default for SoftSAConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200_000,
            initial_temp: 100.0,
            cooling_rate: 0.99995,
            penalty_factor_start: 100.0,
            penalty_factor_end: 1_000_000.0,
            sa_step_scale: 1.0,
        }
    }
}

pub fn optimize_soft_sa(
    mut trees: Vec<Tree>, 
    config: Option<SoftSAConfig>,
    verbose: bool
) -> Vec<Tree> {
    let conf = config.unwrap_or_default();
    let mut rng = rand::thread_rng();
    
    let current_overlap = calculate_overlap_area(&trees);
    let current_kaggle = calculate_kaggle_score(&trees);
    let mut current_energy = current_kaggle + (current_overlap * conf.penalty_factor_start);
    
    let mut temp = conf.initial_temp;
    
    if verbose { println!("🔥 Iniciando Standalone Soft SA..."); }

    for iter in 0..conf.max_iterations {
        let progress = iter as f64 / conf.max_iterations as f64;
        let penalty = conf.penalty_factor_start + (conf.penalty_factor_end - conf.penalty_factor_start) * progress.powi(2);

        let idx = rng.gen_range(0..trees.len());
        let old_x = trees[idx].x;
        let old_y = trees[idx].y;
        let old_angle = trees[idx].angle;
        
        trees[idx].x += rng.gen_range(-0.5..0.5) * conf.sa_step_scale * (temp / conf.initial_temp).sqrt().max(0.01);
        trees[idx].y += rng.gen_range(-0.5..0.5) * conf.sa_step_scale * (temp / conf.initial_temp).sqrt().max(0.01);
        trees[idx].angle += rng.gen_range(-2.0..2.0) * conf.sa_step_scale * (temp / conf.initial_temp).sqrt().max(0.01);
        trees[idx].update_poly();
        
        let new_overlap = calculate_overlap_area(&trees);
        let new_kaggle = calculate_kaggle_score(&trees);
        let new_energy = new_kaggle + (new_overlap * penalty);
        
        if new_energy < current_energy || rng.gen::<f64>() < (-(new_energy - current_energy) / temp).exp() {
            current_energy = new_energy;
        } else {
            trees[idx].x = old_x;
            trees[idx].y = old_y;
            trees[idx].angle = old_angle;
            trees[idx].update_poly();
        }
        
        temp *= conf.cooling_rate;
    }
    
    trees
}

pub struct SoftSARefineConfig {
    pub iterations: usize,
    pub initial_temp: f64,
    pub cooling_rate: f64,
    pub overlap_penalty: f64,
    pub step_scale: f64,
}

impl Default for SoftSARefineConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            initial_temp: 0.1,
            cooling_rate: 0.99,
            overlap_penalty: 10000.0,
            step_scale: 0.5,
        }
    }
}

pub fn refine_soft_sa(
    mut trees: Vec<Tree>,
    config: SoftSARefineConfig,
) -> Vec<Tree> {
    let mut rng = rand::thread_rng();
    let n_trees = trees.len();
    
    let current_overlap = calculate_overlap_area(&trees);
    let current_kaggle = calculate_kaggle_score(&trees);
    let mut current_energy = current_kaggle + (current_overlap * config.overlap_penalty);
    
    let mut best_trees = trees.clone();
    let mut best_energy = current_energy;
    
    let mut temp = config.initial_temp;
    
    for _ in 0..config.iterations {
        let idx = rng.gen_range(0..n_trees);
        let old_x = trees[idx].x;
        let old_y = trees[idx].y;
        let old_angle = trees[idx].angle;
        
        let dx = rng.gen_range(-0.1..0.1) * config.step_scale;
        let dy = rng.gen_range(-0.1..0.1) * config.step_scale;
        let da = rng.gen_range(-2.0..2.0) * config.step_scale;
        
        trees[idx].x += dx;
        trees[idx].y += dy;
        trees[idx].angle += da;
        trees[idx].update_poly();
        
        let new_overlap = calculate_overlap_area(&trees);
        let new_kaggle = calculate_kaggle_score(&trees);
        let new_energy = new_kaggle + (new_overlap * config.overlap_penalty);
        
        let delta = new_energy - current_energy;
        
        if delta < 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
            current_energy = new_energy;
            if current_energy < best_energy {
                best_energy = current_energy;
                best_trees = trees.clone();
            }
        } else {
            trees[idx].x = old_x;
            trees[idx].y = old_y;
            trees[idx].angle = old_angle;
            trees[idx].update_poly();
        }
        
        temp *= config.cooling_rate;
    }
    
    best_trees
}
